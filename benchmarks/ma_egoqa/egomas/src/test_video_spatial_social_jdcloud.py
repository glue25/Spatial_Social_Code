import unittest

from egomas.src.run_video_spatial_social_jdcloud import (
    normalize_video_spatial_code,
    proxy_video_path,
    retrieve_video_ids,
    source_video_path,
)


class VideoSpatialSocialTest(unittest.TestCase):
    def test_video_paths(self) -> None:
        clip = "DAY1_A1_JAKE_12150000.mp4"
        self.assertEqual(
            str(proxy_video_path(__import__("pathlib").Path("proxy"), clip)),
            "proxy/A1_JAKE/DAY1/DAY1_A1_JAKE_12150000.mp4",
        )

    def test_source_path_must_exist(self) -> None:
        with self.assertRaises(FileNotFoundError):
            source_video_path(__import__("pathlib").Path("/missing"), "DAY1_A1_JAKE_12150000.mp4")

    def test_timestamp_strings_become_integer_seconds(self) -> None:
        code = normalize_video_spatial_code(
            'S.event("00:26 - 00:29", "Lucia", "suggests", "dessert")'
        )
        self.assertIn("S.event(26", code)

    def test_option_balanced_retrieval_reserves_each_option(self) -> None:
        class Retriever:
            def retrieve(self, query, top_k, return_scores):
                if "alpha" in query:
                    return ["A.mp4", "shared.mp4"], ["", ""], [20.0, 19.0]
                if "beta" in query:
                    return ["B.mp4", "shared.mp4"], ["", ""], [18.0, 17.0]
                return ["P.mp4"], [""], [16.0]

        clips = retrieve_video_ids(
            Retriever(),
            [{"name": "P", "query": "planner"}],
            top_k_per_agent=1,
            max_videos=3,
            score_threshold=10.0,
            question="question",
            options=["alpha", "beta"],
            top_k_per_option=2,
        )
        self.assertEqual([clip["id"] for clip in clips], ["A.mp4", "B.mp4", "P.mp4"])


if __name__ == "__main__":
    unittest.main()
