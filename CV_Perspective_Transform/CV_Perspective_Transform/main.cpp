#include <opencv2\opencv.hpp>

#define WIDTH 450
#define HEIGHT 600

using namespace std;
using namespace cv;

Mat homography_matrix(Mat x, Mat y, Mat _x, Mat _y) {
	typedef double size;
	Mat A = Mat(8, 1, CV_64FC1);
	
	A.at<size>(0, 0) = _x.at<size>(0, 0);
	A.at<size>(1, 0) = _y.at<size>(0, 0);
	A.at<size>(2, 0) = _x.at<size>(1, 0);
	A.at<size>(3, 0) = _y.at<size>(1, 0);
	A.at<size>(4, 0) = _x.at<size>(2, 0);
	A.at<size>(5, 0) = _y.at<size>(2, 0);
	A.at<size>(6, 0) = _x.at<size>(3, 0);
	A.at<size>(7, 0) = _y.at<size>(3, 0);

	Mat H = Mat::zeros(8, 8, CV_64FC1);
	
	H.at<size>(0, 0) = x.at<size>(0, 0);
	H.at<size>(0, 1) = y.at<size>(0, 0);
	H.at<size>(0, 2) = 1.0;
	H.at<size>(0, 6) = -1.0 * x.at<size>(0, 0) * _x.at<size>(0, 0);
	H.at<size>(0, 7) = -1.0 * _x.at<size>(0, 0) * y.at<size>(0, 0);

	H.at<size>(1, 3) = x.at<size>(0, 0);
	H.at<size>(1, 4) = y.at<size>(0, 0);
	H.at<size>(1, 5) = 1.0;
	H.at<size>(1, 6) = -1.0 * x.at<size>(0, 0) * _y.at<size>(0, 0);
	H.at<size>(1, 7) = -1.0 * y.at<size>(0, 0) * _y.at<size>(0, 0);

	H.at<size>(2, 0) = x.at<size>(1, 0);
	H.at<size>(2, 1) = y.at<size>(1, 0);
	H.at<size>(2, 2) = 1.0;
	H.at<size>(2, 6) = -1.0 * x.at<size>(1, 0) * _x.at<size>(1, 0);
	H.at<size>(2, 7) = -1.0 * _x.at<size>(1, 0) * y.at<size>(1, 0);

	H.at<size>(3, 3) = x.at<size>(1, 0);
	H.at<size>(3, 4) = y.at<size>(1, 0);
	H.at<size>(3, 5) = 1.0;
	H.at<size>(3, 6) = -1.0 * x.at<size>(1, 0) * _y.at<size>(1, 0);
	H.at<size>(3, 7) = -1.0 * y.at<size>(1, 0) * _y.at<size>(1, 0);

	H.at<size>(4, 0) = x.at<size>(2, 0);
	H.at<size>(4, 1) = y.at<size>(2, 0);
	H.at<size>(4, 2) = 1.0;
	H.at<size>(4, 6) = -1.0 * x.at<size>(2, 0) * _x.at<size>(2, 0);
	H.at<size>(4, 7) = -1.0 * _x.at<size>(2, 0) * y.at<size>(2, 0);

	H.at<size>(5, 3) = x.at<size>(2, 0);
	H.at<size>(5, 4) = y.at<size>(2, 0);
	H.at<size>(5, 5) = 1.0;
	H.at<size>(5, 6) = -1.0 * x.at<size>(2, 0) * _y.at<size>(2, 0);
	H.at<size>(5, 7) = -1.0 * y.at<size>(2, 0) * _y.at<size>(2, 0);

	H.at<size>(6, 0) = x.at<size>(3, 0);
	H.at<size>(6, 1) = y.at<size>(3, 0);
	H.at<size>(6, 2) = 1.0;
	H.at<size>(6, 6) = -1.0 * x.at<size>(3, 0) * _x.at<size>(3, 0);
	H.at<size>(6, 7) = -1.0 * _x.at<size>(3, 0) * y.at<size>(3, 0);

	H.at<size>(7, 3) = x.at<size>(3, 0);
	H.at<size>(7, 4) = y.at<size>(3, 0);
	H.at<size>(7, 5) = 1.0;
	H.at<size>(7, 6) = -1.0 * x.at<size>(3, 0) * _y.at<size>(3, 0);
	H.at<size>(7, 7) = -1.0 * y.at<size>(3, 0) * _y.at<size>(3, 0);

	Mat H_inv = H.inv();
	Mat C = H_inv * A;

	Mat homography = Mat(3, 3, CV_64FC1);

	homography.at<size>(0, 0) = C.at<size>(0, 0);
	homography.at<size>(0, 1) = C.at<size>(1, 0);
	homography.at<size>(0, 2) = C.at<size>(2, 0);
	homography.at<size>(1, 0) = C.at<size>(3, 0);
	homography.at<size>(1, 1) = C.at<size>(4, 0);
	homography.at<size>(1, 2) = C.at<size>(5, 0);
	homography.at<size>(2, 0) = C.at<size>(6, 0);
	homography.at<size>(2, 1) = C.at<size>(7, 0);
	homography.at<size>(2, 2) = 1.0;

	return homography;
}

static int bound_check(int x, int y) {
	if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT)
		return 0;
	return 1;
}

void backward_warping(Mat& dst, const Mat& src, Mat& H) {
	typedef double size;
	typedef uchar one_channel;

	one_channel m;

	size w, px, py, p ,q;

	size wx[2];
	size wy[2];

	int i, j, x, y;

	Mat homography_inv = H.inv();

	for (j = 0; j < src.cols; j++) {
		for (i = 0; i < src.rows; i++) {
			w = homography_inv.at<size>(2, 0) * i + homography_inv.at<size>(2, 1) * j + homography_inv.at<size>(2, 2) * 1;

			px = (homography_inv.at<size>(0, 0) * i + homography_inv.at<size>(0, 1) * j + homography_inv.at<size>(0, 2)) / w;
			py = (homography_inv.at<size>(1, 0) * i + homography_inv.at<size>(1, 1) * j + homography_inv.at<size>(1, 2)) / w;

			wx[1] = px - floor(px);
			wx[0] = 1.0 - wx[1];

			wy[1] = py - floor(py);
			wy[0] = 1.0 - wy[1];

			x = floor(px);
			y = floor(py);

			if (bound_check(x, y) && bound_check(x, y + 1)) {
				p = wy[1] * src.at<one_channel>(x, y + 1) + wy[0] * src.at<one_channel>(x, y);
			}
			if (bound_check(x + 1, y) && bound_check(x + 1, y + 1)) {
				q = wy[1] * src.at<one_channel>(x + 1, y) + wy[0] * src.at<one_channel>(x + 1, y + 1);
			}

			m = (uchar)(p * wx[0] + q * wx[1]);

			dst.at<one_channel>(i, j) = m;
		}
	}
}

int main() {
	typedef double size;
	Mat src = imread("C:\\Users\\hyeonjun\\Desktop\\4-2\\영상화질개선\\input.jpg", 0);
	//Mat src;

	//resize(input, src, Size(450, 600));

	Mat dst = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	//imshow("dst", src);
	//waitKey(0);
	
	// 그림판에서는 (x, y)순서이지만 opencv c++에서는 (y, x)순서
	Mat x = Mat_<size>({ 4, 1 }, { 152, 179, 435, 385 }); // 그림판 좌표 left_top(147, 152)
	Mat y = Mat_<size>({ 4, 1 }, { 147, 324, 171, 342 }); // 그림판 좌표 right_top(324, 179)
	Mat _x = Mat_<size>({ 4, 1 }, { 0, 0, 600, 600 });	  // 그림판 좌표 left_bottom(171, 435)
	Mat _y = Mat_<size>({ 4, 1 }, { 0, 450, 0, 450 });	  // 그림판 좌표 right_bottom(342, 385)

	Mat H = homography_matrix(x, y, _x, _y);
	backward_warping(dst, src, H);

	imshow("perspective transform", dst);
	waitKey(0);

	return 0;
}