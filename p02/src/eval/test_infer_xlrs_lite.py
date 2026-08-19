import unittest

from PIL import Image

from infer_xlrs_lite import (
    ANSWER_FORMAT_BY_CATEGORY,
    ANSWER_FORMAT_MULTI_4,
    ANSWER_FORMAT_SINGLE_2,
    ANSWER_FORMAT_SINGLE_4,
    IMAGE_TOKEN,
    TASK_PAIRS,
    answer_protocol,
    answer_format_for_category,
    answer_format,
    answer_obeys_protocol,
    clip_bbox_to_image,
    crop_box_for_image,
    evenly_spaced_offsets,
    extract_reference_bbox,
    extract_answer_payload,
    make_stage1_prompt,
    make_stage2_prompt,
    map_bbox_between_images,
    parse_positions,
    question_for_global_image,
    question_with_reference_bbox,
    reference_bbox_for_image,
    score_records,
    segmented_vision_attention_forward,
    xlrs_doc_to_text,
)


class InferXLRSAdapterTests(unittest.TestCase):
    FOUR_CHOICE_CATEGORY = "Object properties/Object color"
    TWO_CHOICE_CATEGORY = "Object properties/Object motion state"
    MULTI_CHOICE_CATEGORY = (
        "Land use classification/Overall Land use classification"
    )

    @staticmethod
    def sample(category, option_count=4):
        options = [
            f"({chr(ord('A') + index)}) option {index + 1}"
            for index in range(option_count)
        ]
        return {"category": category, "multi-choice options": options}

    def test_xlrs_answer_constraint_is_last_user_instruction(self):
        sample = self.sample(self.FOUR_CHOICE_CATEGORY)
        prompt = make_stage1_prompt("question", 2, sample)

        self.assertEqual(prompt.count(IMAGE_TOKEN), 2)
        self.assertTrue(prompt.endswith("<|im_end|><|im_start|>assistant\n"))
        self.assertIn(
            "strictly following the format requirements in the XLRS answer format below",
            prompt,
        )
        self.assertIn("output exactly one uppercase letter from ABCD", prompt)
        self.assertGreater(
            prompt.rfind("Inside <answer>"), prompt.rfind("strictly following")
        )

    def test_every_category_has_exactly_one_output_protocol(self):
        self.assertEqual(set(ANSWER_FORMAT_BY_CATEGORY), set(TASK_PAIRS))
        self.assertEqual(
            answer_format_for_category(self.FOUR_CHOICE_CATEGORY),
            ANSWER_FORMAT_SINGLE_4,
        )
        self.assertEqual(
            answer_format_for_category(self.TWO_CHOICE_CATEGORY),
            ANSWER_FORMAT_SINGLE_2,
        )
        self.assertEqual(
            answer_format_for_category(self.MULTI_CHOICE_CATEGORY),
            ANSWER_FORMAT_MULTI_4,
        )

    def test_prompt_contains_only_the_active_answer_protocol(self):
        binary_sample = self.sample(self.TWO_CHOICE_CATEGORY, option_count=2)
        binary_prompt = make_stage1_prompt("question", 1, binary_sample)
        self.assertIn("output exactly one uppercase letter from AB", binary_prompt)
        self.assertNotIn("from ABCD", binary_prompt)
        self.assertNotIn("Select every option that applies", binary_prompt)

        multi_sample = self.sample(self.MULTI_CHOICE_CATEGORY)
        multi_prompt = make_stage1_prompt("question", 1, multi_sample)
        self.assertIn("applicable uppercase letters from ABCD", multi_prompt)
        self.assertIn("with no spaces or separators", multi_prompt)
        self.assertNotIn("output exactly one uppercase letter", multi_prompt)

    def test_answer_protocol_text_matches_ver1(self):
        sample = self.sample(self.FOUR_CHOICE_CATEGORY)

        self.assertEqual(
            answer_protocol(sample),
            "\nXLRS answer format:\n"
            "Select exactly one best option. Inside <answer>, output exactly one "
            "uppercase letter from ABCD, with no other text.\n",
        )

    def test_question_prompt_has_no_generic_answer_format(self):
        doc = {
            "question": "Is the object moving?",
            "multi-choice options": ["(A) Yes", "(B) No"],
            "category": self.TWO_CHOICE_CATEGORY,
        }
        prompt = xlrs_doc_to_text(doc)

        self.assertEqual(
            prompt,
            "Is the object moving?\n\nThe choices are listed below:\n"
            "(A) Yes\n(B) No",
        )
        self.assertNotIn("Only respond", prompt)
        self.assertNotIn("Select exactly one", prompt)

    def test_unknown_category_has_no_fallback_protocol(self):
        with self.assertRaises(ValueError):
            answer_format_for_category("unknown/category")

    def test_answer_format(self):
        cases = [
            ("A", "letters"),
            ("A C", "letters"),
            ("White", "semantic"),
            ("2", "semantic"),
            (None, "missing"),
        ]
        for answer, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(answer_format(answer), expected)

    def test_extract_answer_payload_accepts_wrapped_or_bare_letters_only(self):
        self.assertEqual(extract_answer_payload("<answer>A</answer>"), "A")
        self.assertEqual(extract_answer_payload("BC"), "BC")
        self.assertIsNone(extract_answer_payload("The answer is A"))

    def test_answer_protocol_compliance_is_category_specific(self):
        cases = [
            ("A", self.FOUR_CHOICE_CATEGORY, True),
            ("AB", self.FOUR_CHOICE_CATEGORY, False),
            ("B", self.TWO_CHOICE_CATEGORY, True),
            ("C", self.TWO_CHOICE_CATEGORY, False),
            ("ACD", self.MULTI_CHOICE_CATEGORY, True),
            ("DCA", self.MULTI_CHOICE_CATEGORY, False),
            ("A C", self.MULTI_CHOICE_CATEGORY, False),
            ("AA", self.MULTI_CHOICE_CATEGORY, False),
            (None, self.FOUR_CHOICE_CATEGORY, False),
        ]
        for answer, category, expected in cases:
            with self.subTest(answer=answer, category=category):
                self.assertEqual(
                    answer_obeys_protocol(answer, category), expected
                )

    def test_parse_positions_supports_ranges_and_deduplicates(self):
        self.assertEqual(parse_positions("3-5,4,8", 10), [3, 4, 5, 8])

    def test_parse_positions_rejects_invalid_ranges(self):
        with self.assertRaises(ValueError):
            parse_positions("5-3", 10)
        with self.assertRaises(ValueError):
            parse_positions("10", 10)

    def test_evenly_spaced_offsets_match_balanced_selection_rule(self):
        self.assertEqual(evenly_spaced_offsets(10, 3), [0, 4, 8])
        self.assertEqual(evenly_spaced_offsets(10, 2), [0, 9])
        self.assertEqual(evenly_spaced_offsets(3, 3), [0, 1, 2])

    def test_weighted_accuracy_uses_full_dataset_category_counts(self):
        records = []
        for position, category in enumerate(TASK_PAIRS):
            correct = category == "Object properties/Object classification"
            records.append(
                {
                    "dataset_position": position,
                    "category": category,
                    "answer": "A",
                    "prediction": "A" if correct else "B",
                    "stage1_prediction": "B",
                    "status": "ok",
                }
            )

        metrics = score_records(records)

        self.assertAlmostEqual(
            metrics["category_count_weighted_accuracy"], 800 / 3080
        )

    def test_segmented_vision_attention_matches_block_diagonal_attention(self):
        from transformers.models.qwen2_5_vl import (
            Qwen2_5_VLVisionConfig,
            modeling_qwen2_5_vl,
        )

        torch = __import__("torch")
        torch.manual_seed(7)
        attention_class = getattr(
            modeling_qwen2_5_vl,
            "Qwen2_5_VLVisionSdpaAttention",
            modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention,
        )
        if attention_class is modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention:
            config = Qwen2_5_VLVisionConfig(
                hidden_size=32, num_heads=4, intermediate_size=64
            )
            attention = attention_class(config).eval()
        else:
            attention = attention_class(dim=32, num_heads=4).eval()
        hidden_states = torch.randn(6, 32)
        cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
        position_embeddings = (torch.ones(6, 8), torch.zeros(6, 8))

        expected = attention(
            hidden_states,
            cu_seqlens,
            position_embeddings=position_embeddings,
        )
        actual = segmented_vision_attention_forward(
            attention,
            hidden_states,
            cu_seqlens,
            position_embeddings=position_embeddings,
        )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_reference_bbox_is_mapped_from_question_resolution(self):
        question = (
            "Image resolution: 7360 x 4912. "
            "Bounding box: [894, 2080, 935, 2137]."
        )
        reference = extract_reference_bbox(question)

        self.assertEqual(reference, (7360.0, 4912.0, [894.0, 2080.0, 935.0, 2137.0]))
        mapped = reference_bbox_for_image(reference, Image.new("RGB", (3680, 2456)))
        self.assertEqual(mapped, [447.0, 1040.0, 467.5, 1068.5])

    def test_reference_question_uses_stage1_global_image_coordinates(self):
        question = (
            "Determine the color. Image resolution: 7360 x 4912. "
            "Bounding box: [894, 2080, 935, 2137]."
        )
        global_image = Image.new("RGB", (1024, 683))

        rewritten = question_for_global_image(question, global_image)
        rewritten_reference = extract_reference_bbox(rewritten)

        self.assertNotIn("7360 x 4912", rewritten)
        self.assertIn("Image resolution: 1024 x 683. Bounding box:", rewritten)
        self.assertEqual(rewritten_reference[:2], (1024.0, 683.0))
        expected = reference_bbox_for_image(
            extract_reference_bbox(question), global_image
        )
        for actual, target in zip(rewritten_reference[2], expected):
            self.assertAlmostEqual(actual, target, places=5)

    def test_global_model_bbox_maps_back_with_exact_axis_scales(self):
        global_image = Image.new("RGB", (1024, 683))
        original_image = Image.new("RGB", (7360, 4912))
        reference = (7360.0, 4912.0, [894.0, 2080.0, 935.0, 2137.0])
        bbox = reference_bbox_for_image(reference, global_image)

        mapped = map_bbox_between_images(bbox, global_image, original_image)

        expected = reference[2]
        for actual, target in zip(mapped, expected):
            self.assertAlmostEqual(actual, target, places=3)

    def test_crop_box_and_local_reference_coordinates_are_consistent(self):
        original = Image.new("RGB", (10000, 10000))
        reference_bbox = [3373.0, 4097.0, 3429.0, 4146.0]

        crop_box = crop_box_for_image(original, reference_bbox, min_size=512)
        crop = original.crop(tuple(crop_box))
        local_bbox = [
            reference_bbox[0] - crop_box[0],
            reference_bbox[1] - crop_box[1],
            reference_bbox[2] - crop_box[0],
            reference_bbox[3] - crop_box[1],
        ]
        rewritten = question_with_reference_bbox(
            "Image resolution: 10000 x 10000. "
            "Bounding box: [3373, 4097, 3429, 4146].",
            crop,
            local_bbox,
        )

        self.assertEqual(crop_box, [3145, 3865, 3657, 4377])
        self.assertEqual(
            extract_reference_bbox(rewritten),
            (512.0, 512.0, [228.0, 232.0, 284.0, 281.0]),
        )

    def test_crop_local_bbox_is_clipped_for_anisotropic_reference(self):
        crop = Image.new("RGB", (512, 512))

        clipped = clip_bbox_to_image([36.0, -105.0, 476.0, 617.0], crop)

        self.assertEqual(clipped, [36.0, 0.0, 476.0, 512.0])

    def test_reference_stage2_prompt_reuses_stage1_context(self):
        sample = self.sample(self.FOUR_CHOICE_CATEGORY)
        stage1_prompt = make_stage1_prompt(
            "question with reference coordinates", 1, sample
        )
        stage1_output = (
            '<think>locate target</think>'
            '<stage_2_reasoning>[{"bbox_2d": [1, 2, 3, 4]}]</stage_2_reasoning>'
            '<answer>A</answer>'
        )
        prompt = make_stage2_prompt(stage1_prompt, stage1_output, 1)

        self.assertEqual(prompt.count(IMAGE_TOKEN), 2)
        self.assertTrue(prompt.startswith(stage1_prompt))
        self.assertIn("question with reference coordinates", prompt)
        self.assertIn("<think>locate target</think>", prompt)
        self.assertIn('"bbox_2d": [1, 2, 3, 4]', prompt)
        self.assertNotIn("<answer>A</answer>", prompt)
        self.assertIn("output exactly one uppercase letter from ABCD", prompt)
        self.assertTrue(prompt.endswith(IMAGE_TOKEN))

    def test_reference_bbox_is_absent_for_non_reference_questions(self):
        self.assertIsNone(extract_reference_bbox("How many buildings are visible?"))


if __name__ == "__main__":
    unittest.main()
