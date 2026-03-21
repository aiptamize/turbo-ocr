#include <catch_amalgamated.hpp>

#include "turbo_ocr/detection/det_postprocess.h"
#include <opencv2/imgproc.hpp>

using turbo_ocr::Box;
using turbo_ocr::detection::box_score_fast;
using turbo_ocr::detection::get_mini_boxes;
using turbo_ocr::detection::unclip;

TEST_CASE("get_mini_boxes returns ordered corners", "[det_postprocess]") {
  // A simple rectangle contour
  std::vector<cv::Point> contour = {{10, 10}, {50, 10}, {50, 30}, {10, 30}};
  float min_side = 0;
  Box box = get_mini_boxes(contour, min_side);

  // min_side should be the shorter dimension (height=20)
  CHECK(min_side == Catch::Approx(20.0f).margin(1.0f));

  // top-left should have smallest y among left pair, smallest x among top pair
  // Verify ordering: tl.y <= bl.y, tr.y <= br.y, tl.x <= tr.x
  CHECK(box[0][1] <= box[3][1]); // tl.y <= bl.y
  CHECK(box[1][1] <= box[2][1]); // tr.y <= br.y
  CHECK(box[0][0] <= box[1][0]); // tl.x <= tr.x
}

TEST_CASE("get_mini_boxes handles tilted contour", "[det_postprocess]") {
  // Slightly rotated rectangle
  std::vector<cv::Point> contour = {{15, 5}, {55, 10}, {53, 35}, {13, 30}};
  float min_side = 0;
  Box box = get_mini_boxes(contour, min_side);

  // Should still produce a valid 4-corner box
  CHECK(min_side > 0);
  // All corners should be close to the input contour bounding region
  for (int i = 0; i < 4; ++i) {
    CHECK(box[i][0] >= 0);
    CHECK(box[i][1] >= 0);
  }
}

TEST_CASE("unclip expands polygon", "[det_postprocess]") {
  std::vector<cv::Point> polygon = {{10, 10}, {50, 10}, {50, 30}, {10, 30}};
  float unclip_ratio = 1.5f;
  auto expanded = unclip(polygon, unclip_ratio);

  // Expanded polygon should have at least 3 points
  REQUIRE(expanded.size() >= 3);

  // The bounding rect of the expanded polygon should be larger
  cv::Rect orig_br = cv::boundingRect(polygon);
  cv::Rect exp_br = cv::boundingRect(expanded);
  CHECK(exp_br.width >= orig_br.width);
  CHECK(exp_br.height >= orig_br.height);
}

TEST_CASE("unclip with zero perimeter returns original", "[det_postprocess]") {
  // Degenerate polygon (single point repeated)
  std::vector<cv::Point> polygon = {{10, 10}, {10, 10}, {10, 10}};
  auto result = unclip(polygon, 1.5f);
  // Should return original (no crash)
  CHECK(result.size() == polygon.size());
}

TEST_CASE("box_score_fast computes mean within polygon", "[det_postprocess]") {
  // Create a small prediction map filled with 0.8
  cv::Mat pred_map(100, 100, CV_32F, cv::Scalar(0.8f));

  // A rectangle covering part of the image
  std::vector<cv::Point> contour = {{20, 20}, {60, 20}, {60, 50}, {20, 50}};

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);

  // Should be approximately 0.8 (uniform fill)
  CHECK(score == Catch::Approx(0.8f).margin(0.01f));
}

TEST_CASE("box_score_fast returns zero for out-of-bounds contour", "[det_postprocess]") {
  cv::Mat pred_map(50, 50, CV_32F, cv::Scalar(0.9f));

  // Contour outside image bounds (negative coords clamped to 0)
  // All points at origin => xmax <= xmin => returns 0
  std::vector<cv::Point> contour = {{0, 0}, {0, 0}, {0, 0}};

  std::vector<cv::Point> shifted_buf;
  cv::Mat mask_buf;
  float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);

  CHECK(score == Catch::Approx(0.0f).margin(0.01f));
}
