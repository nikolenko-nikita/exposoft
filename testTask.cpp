/*
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Jie Zhang(a Ph.D supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/

#include<iostream>
using namespace std;

#ifdef _WIN32
#pragma once
#include <opencv2/core/version.hpp>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) \
	CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif //_DEBUG

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )

#endif //_WIN32

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include "math_functions.h"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace seeta;

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "./data/";
std::string MODEL_DIR = "./model/";
#endif

std::string SAVE_PATH = "results/";
std::string TRAIN_TEST_PATH = "C:/exposoft/FR_test/lfw/";

void testTask(int argc, char* argv[]){

	if (argc != 3){
		cout << "Usage: " << argv[0]
			<< " image1_path image2_path"
			<< endl;
		return;
	}

	// Initialize face detection model
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("../../../FaceAlignment/model/seeta_fa_v1.1.bin");

	// Initialize face Identification model 
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
	std::string test_dir = DATA_DIR + "test_face_recognizer/";

	//load image
	cv::Mat first_img_color = cv::imread(test_dir + argv[1], 1);
	cv::Mat imgGray;
	cv::cvtColor(first_img_color, imgGray, CV_BGR2GRAY);

	cv::Mat second_img_color = cv::imread(test_dir + argv[2], 1);
	cv::Mat second_img_gray;
	cv::cvtColor(second_img_color, second_img_gray, CV_BGR2GRAY);

	ImageData first_img_data_color(first_img_color.cols, first_img_color.rows, first_img_color.channels());
	first_img_data_color.data = first_img_color.data;

	ImageData first_img_data_gray(imgGray.cols, imgGray.rows, imgGray.channels());
	first_img_data_gray.data = imgGray.data;

	ImageData second_img_data_color(second_img_color.cols, second_img_color.rows, second_img_color.channels());
	second_img_data_color.data = second_img_color.data;

	ImageData second_img_data_gray(second_img_gray.cols, second_img_gray.rows, second_img_gray.channels());
	second_img_data_gray.data = second_img_gray.data;

	// Detect faces
	std::vector<seeta::FaceInfo> first_faces = detector.Detect(first_img_data_gray);
	int32_t first_face_num = static_cast<int32_t>(first_faces.size());

	std::vector<seeta::FaceInfo> second_faces = detector.Detect(second_img_data_gray);
	int32_t second_face_num = static_cast<int32_t>(second_faces.size());

	if (first_face_num == 0 || second_face_num == 0)
	{
		std::cout << "Faces are not detected.";
		return;
	}

	// Detect 5 facial landmarks
	seeta::FacialLandmark first_points[5];
	point_detector.PointDetectLandmarks(first_img_data_gray, first_faces[0], first_points);

	seeta::FacialLandmark second_points[5];
	point_detector.PointDetectLandmarks(second_img_data_gray, second_faces[0], second_points);

	//Clone image for save them with face points
	cv::Mat first_img_color_point = first_img_color.clone();
	cv::Mat second_img_color_point = second_img_color.clone();

	for (int i = 0; i<5; i++)
	{
		cv::circle(first_img_color_point, cv::Point(first_points[i].x, first_points[i].y), 2,
			CV_RGB(0, 255, 0));
		cv::circle(second_img_color_point, cv::Point(second_points[i].x, second_points[i].y), 2,
			CV_RGB(0, 255, 0));
	}
	cv::imwrite("points_" + string(argv[1]), first_img_color_point);
	cv::imwrite("points_" + string(argv[2]), second_img_color_point);

	// Create a images to store crop face.
	cv::Mat first_crop_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
	ImageData first_crop_img_data(first_crop_img.cols, first_crop_img.rows, first_crop_img.channels());
	first_crop_img_data.data = first_crop_img.data;

	cv::Mat second_crop_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
	ImageData second_crop_img_data(second_crop_img.cols, second_crop_img.rows, second_crop_img.channels());
	second_crop_img_data.data = second_crop_img.data;

	face_recognizer.CropFace(first_img_data_color, first_points, first_crop_img_data);
	face_recognizer.CropFace(second_img_data_color, second_points, second_crop_img_data);

	cv::imwrite("crop_" + string(argv[1]), first_crop_img);
	cv::imwrite("crop_" + string(argv[2]), second_crop_img);

	// Extract face identity feature
	float first_fea[2048];
	float second_fea[2048];
	face_recognizer.ExtractFeatureWithCrop(first_img_data_color, first_points, first_fea);
	face_recognizer.ExtractFeatureWithCrop(second_img_data_color, second_points, second_fea);
	
	// Caculate similarity of two faces
	float sim = face_recognizer.CalcSimilarity(first_fea, second_fea);
	std::cout << sim << endl;

	// Extract face identity feature
	float first_fea1[2048];
	float second_fea1[2048];
	face_recognizer.ExtractFeature(first_crop_img_data, first_fea1);
	face_recognizer.ExtractFeature(second_crop_img_data, second_fea1);

	// Caculate similarity of two faces
	float sim1 = face_recognizer.CalcSimilarity(first_fea1, second_fea1);
	std::cout << sim1 << endl;

	return;

}

void testTaskFileList(int argc, char* argv[]){

	if (argc != 2){
		cout << "Usage: " << argv[0]
			<< " image_list_file"
			<< endl;
		return;
	}

	// Initialize face detection model
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("../../../FaceAlignment/model/seeta_fa_v1.1.bin");

	// Initialize face Identification model 
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
	//std::string test_dir = DATA_DIR + "test_face_recognizer/";

	std::ifstream ifs;
	ifs.open(argv[1], std::ifstream::in);

	std::string first_image_path, second_image_path;
	
	while (ifs >> first_image_path, ifs >> second_image_path){

		std::string first_image_name, second_image_name;

		first_image_name = first_image_path.substr(first_image_path.find_last_of('/') + 1);
		second_image_name = second_image_path.substr(second_image_path.find_last_of('/') + 1);

		//load image
		cv::Mat first_img_color = cv::imread(first_image_path, 1);
		cv::Mat imgGray;
		cv::cvtColor(first_img_color, imgGray, CV_BGR2GRAY);

		cv::Mat second_img_color = cv::imread(second_image_path, 1);
		cv::Mat second_img_gray;
		cv::cvtColor(second_img_color, second_img_gray, CV_BGR2GRAY);

		ImageData first_img_data_color(first_img_color.cols, first_img_color.rows, first_img_color.channels());
		first_img_data_color.data = first_img_color.data;

		ImageData first_img_data_gray(imgGray.cols, imgGray.rows, imgGray.channels());
		first_img_data_gray.data = imgGray.data;

		ImageData second_img_data_color(second_img_color.cols, second_img_color.rows, second_img_color.channels());
		second_img_data_color.data = second_img_color.data;

		ImageData second_img_data_gray(second_img_gray.cols, second_img_gray.rows, second_img_gray.channels());
		second_img_data_gray.data = second_img_gray.data;

		// Detect faces
		std::vector<seeta::FaceInfo> first_faces = detector.Detect(first_img_data_gray);
		int32_t first_face_num = static_cast<int32_t>(first_faces.size());

		std::vector<seeta::FaceInfo> second_faces = detector.Detect(second_img_data_gray);
		int32_t second_face_num = static_cast<int32_t>(second_faces.size());

		if (first_face_num == 0 || second_face_num == 0)
		{
			if (first_face_num == 0)
			{
				std::cout << first_image_path << ": " << "Face are not detected." << endl;
			}
			if (second_face_num == 0)
			{
				std::cout << second_image_path << ": " << "Face are not detected." << endl;
			}
			continue;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark first_points[5];
		point_detector.PointDetectLandmarks(first_img_data_gray, first_faces[0], first_points);

		seeta::FacialLandmark second_points[5];
		point_detector.PointDetectLandmarks(second_img_data_gray, second_faces[0], second_points);

		//Clone image for save them with face points
		cv::Mat first_img_color_point = first_img_color.clone();
		cv::Mat second_img_color_point = second_img_color.clone();

		for (int i = 0; i < 5; i++)
		{
			cv::circle(first_img_color_point, cv::Point(first_points[i].x, first_points[i].y), 2,
				CV_RGB(0, 255, 0));
			cv::circle(second_img_color_point, cv::Point(second_points[i].x, second_points[i].y), 2,
				CV_RGB(0, 255, 0));
		}
				
		cv::imwrite(SAVE_PATH + "points_" + first_image_name, first_img_color_point);
		cv::imwrite(SAVE_PATH + "points_" + second_image_name, second_img_color_point);

		// Create a images to store crop face.
		cv::Mat first_crop_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
		ImageData first_crop_img_data(first_crop_img.cols, first_crop_img.rows, first_crop_img.channels());
		first_crop_img_data.data = first_crop_img.data;

		cv::Mat second_crop_img(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
		ImageData second_crop_img_data(second_crop_img.cols, second_crop_img.rows, second_crop_img.channels());
		second_crop_img_data.data = second_crop_img.data;

		face_recognizer.CropFace(first_img_data_color, first_points, first_crop_img_data);
		face_recognizer.CropFace(second_img_data_color, second_points, second_crop_img_data);

		cv::imwrite(SAVE_PATH + "crop_" + first_image_name, first_crop_img);
		cv::imwrite(SAVE_PATH + "crop_" + second_image_name, second_crop_img);

		// Extract face identity feature
		float first_fea[2048];
		float second_fea[2048];
		face_recognizer.ExtractFeatureWithCrop(first_img_data_color, first_points, first_fea);
		face_recognizer.ExtractFeatureWithCrop(second_img_data_color, second_points, second_fea);

		// Caculate similarity of two faces
		float sim = face_recognizer.CalcSimilarity(first_fea, second_fea);
		std::cout << sim << endl;

		//// Extract face identity feature
		//float first_fea1[2048];
		//float second_fea1[2048];
		//face_recognizer.ExtractFeature(first_crop_img_data, first_fea1);
		//face_recognizer.ExtractFeature(second_crop_img_data, second_fea1);

		//// Caculate similarity of two faces
		//float sim1 = face_recognizer.CalcSimilarity(first_fea1, second_fea1);
		//std::cout << sim1 << endl;
	}

	return;

}

void trainTest(int argc, char* argv[]){

	if (argc != 2){
		cout << "Usage: " << argv[0]
			<< " image_list_file"
			<< endl;
		return;
	}

	// Initialize face detection model
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("../../../FaceAlignment/model/seeta_fa_v1.1.bin");

	// Initialize face Identification model 
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
	//std::string test_dir = DATA_DIR + "test_face_recognizer/";

	std::ifstream ifs;
	std::ofstream ofs;

	ifs.open(argv[1], std::ifstream::in);
	ofs.open("outputs.txt");

	std::string first_image_path, second_image_path;

	int pairs;
	ifs >> pairs;

	//float* score = new float[2 * pairs];
	std::vector<float> score;
	//float* labels = new float[2 * pairs];
	std::vector<float> labels;

	for (int i = 0; i < pairs; i++){

		std::string image_name;
		std::string first_image_number, second_image_number;

		ifs >> image_name;
		ifs >> first_image_number;
		ifs >> second_image_number;

		std::string first_image_path, second_image_path;

		first_image_path = TRAIN_TEST_PATH + image_name + "/" + image_name + "_" + string(4 - first_image_number.length(), '0') + first_image_number + ".jpg";
		second_image_path = TRAIN_TEST_PATH + image_name + "/" + image_name + "_" + string(4 - second_image_number.length(), '0') + second_image_number + ".jpg";

		//load image
		cv::Mat first_img_color = cv::imread(first_image_path, 1);
		cv::Mat imgGray;
		cv::cvtColor(first_img_color, imgGray, CV_BGR2GRAY);

		cv::Mat second_img_color = cv::imread(second_image_path, 1);
		cv::Mat second_img_gray;
		cv::cvtColor(second_img_color, second_img_gray, CV_BGR2GRAY);

		ImageData first_img_data_color(first_img_color.cols, first_img_color.rows, first_img_color.channels());
		first_img_data_color.data = first_img_color.data;

		ImageData first_img_data_gray(imgGray.cols, imgGray.rows, imgGray.channels());
		first_img_data_gray.data = imgGray.data;

		ImageData second_img_data_color(second_img_color.cols, second_img_color.rows, second_img_color.channels());
		second_img_data_color.data = second_img_color.data;

		ImageData second_img_data_gray(second_img_gray.cols, second_img_gray.rows, second_img_gray.channels());
		second_img_data_gray.data = second_img_gray.data;

		// Detect faces
		std::vector<seeta::FaceInfo> first_faces = detector.Detect(first_img_data_gray);
		int32_t first_face_num = static_cast<int32_t>(first_faces.size());

		std::vector<seeta::FaceInfo> second_faces = detector.Detect(second_img_data_gray);
		int32_t second_face_num = static_cast<int32_t>(second_faces.size());

		if (first_face_num == 0 || second_face_num == 0)
		{
			if (first_face_num == 0)
			{
				std::cout << first_image_path << ": " << "Face are not detected." << endl;
			}
			if (second_face_num == 0)
			{
				std::cout << second_image_path << ": " << "Face are not detected." << endl;
			}
			continue;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark first_points[5];
		point_detector.PointDetectLandmarks(first_img_data_gray, first_faces[0], first_points);

		seeta::FacialLandmark second_points[5];
		point_detector.PointDetectLandmarks(second_img_data_gray, second_faces[0], second_points);

		// Extract face identity feature
		float* first_fea = new float[2048];
		float* second_fea = new float[2048];
		face_recognizer.ExtractFeatureWithCrop(first_img_data_color, first_points, first_fea);
		face_recognizer.ExtractFeatureWithCrop(second_img_data_color, second_points, second_fea);

		// Caculate similarity of two faces
		float sim = face_recognizer.CalcSimilarity(first_fea, second_fea);
		//std::cout << sim << endl;

		score.push_back(sim);
		labels.push_back(1.0f);
		delete first_fea;
		delete second_fea;
	}

	for (int i = 0; i < pairs; i++){

		std::string first_image_name, second_image_name;
		std::string first_image_number, second_image_number;

		ifs >> first_image_name;
		ifs >> first_image_number;
		ifs >> second_image_name;
		ifs >> second_image_number;

		std::string first_image_path, second_image_path;

		first_image_path = TRAIN_TEST_PATH + first_image_name + "/" + first_image_name + "_" + string(4 - first_image_number.length(), '0') + first_image_number + ".jpg";
		second_image_path = TRAIN_TEST_PATH + second_image_name + "/" + second_image_name + "_" + string(4 - second_image_number.length(), '0') + second_image_number + ".jpg";

		//load image
		cv::Mat first_img_color = cv::imread(first_image_path, 1);
		cv::Mat imgGray;
		cv::cvtColor(first_img_color, imgGray, CV_BGR2GRAY);

		cv::Mat second_img_color = cv::imread(second_image_path, 1);
		cv::Mat second_img_gray;
		cv::cvtColor(second_img_color, second_img_gray, CV_BGR2GRAY);

		ImageData first_img_data_color(first_img_color.cols, first_img_color.rows, first_img_color.channels());
		first_img_data_color.data = first_img_color.data;

		ImageData first_img_data_gray(imgGray.cols, imgGray.rows, imgGray.channels());
		first_img_data_gray.data = imgGray.data;

		ImageData second_img_data_color(second_img_color.cols, second_img_color.rows, second_img_color.channels());
		second_img_data_color.data = second_img_color.data;

		ImageData second_img_data_gray(second_img_gray.cols, second_img_gray.rows, second_img_gray.channels());
		second_img_data_gray.data = second_img_gray.data;

		// Detect faces
		std::vector<seeta::FaceInfo> first_faces = detector.Detect(first_img_data_gray);
		int32_t first_face_num = static_cast<int32_t>(first_faces.size());

		std::vector<seeta::FaceInfo> second_faces = detector.Detect(second_img_data_gray);
		int32_t second_face_num = static_cast<int32_t>(second_faces.size());

		if (first_face_num == 0 || second_face_num == 0)
		{
			if (first_face_num == 0)
			{
				std::cout << first_image_path << ": " << "Face are not detected." << endl;
			}
			if (second_face_num == 0)
			{
				std::cout << second_image_path << ": " << "Face are not detected." << endl;
			}
			continue;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark first_points[5];
		point_detector.PointDetectLandmarks(first_img_data_gray, first_faces[0], first_points);

		seeta::FacialLandmark second_points[5];
		point_detector.PointDetectLandmarks(second_img_data_gray, second_faces[0], second_points);

		// Extract face identity feature
		float* first_fea = new float[2048];
		float* second_fea = new float[2048];
		face_recognizer.ExtractFeatureWithCrop(first_img_data_color, first_points, first_fea);
		face_recognizer.ExtractFeatureWithCrop(second_img_data_color, second_points, second_fea);

		// Caculate similarity of two faces
		float sim = face_recognizer.CalcSimilarity(first_fea, second_fea);
		//std::cout << sim << endl;

		score.push_back(sim);
		labels.push_back(0.0f);
		delete first_fea;
		delete second_fea;
	}

	for (int i = 0; i < score.size(); i++){
		ofs << score[i] << " " << labels[i] << endl;
	}

	int N = 100;

	float score_max = 0.0f, score_min = 1.0f;

	for (int i = 0; i < score.size(); i++){

		if (score[i] < score_min)
			score_min = score[i];

		if (score[i] > score_max)
			score_max = score[i];
	}

	cout << score_min << " " << score_max << endl;

	float step = (score_max - score_min) / (N - 1);
	float* thresh = new float[N];
	float* accuracy = new float[N];

	for (int i = 0; i < N; i++){
		thresh[i] = score_min + step*i;
		accuracy[i] = 0;
	}

	float M = 0.0f, M_thresh = 0.0f;
	float eps = 0.001;

	for (int i = 0; i < N; i++){
		
		float* res = new float[score.size()];

		for (int j = 0; j < score.size(); j++){

			if (score[j] < thresh[i])
				res[j] = 1.0f;
			else
				res[j] = 0.0f;
		}

		int right_ans = 0;

		for (int j = 0; j < score.size(); j++){

			if (abs(labels[j] - res[j]) < eps)
				right_ans++;
		}

		accuracy[i] = ((float)right_ans) / (score.size());
		if (accuracy[i] > M){
			M_thresh = thresh[i];
			M = accuracy[i];
		}

		delete res;
	}
	
	cout << M << " " << M_thresh << endl;

	delete thresh;
	ofs.close();
	ifs.close();
	return;

}

void threshTest(){

	std::string inputFilePath = "C:/exposoft/FR_test/pairsDevTest.txt";
	std::string outputSimFilePath = "C:/exposoft/FR_test/pairsDevTestOutputCosDistance.txt";
	std::string filesDirectory = "C:/exposoft/FR_test/lfw/";
	float M_thresh_test = 0.5466880f;

	// Initialize face detection model
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("../../../FaceAlignment/model/seeta_fa_v1.1.bin");

	// Initialize face Identification model 
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());
	//std::string test_dir = DATA_DIR + "test_face_recognizer/";

	std::ifstream ifs;
	std::ofstream ofs;

	ifs.open(inputFilePath, std::ifstream::in);
	ofs.open(outputSimFilePath);

	std::string first_image_path, second_image_path;

	int pairs;
	ifs >> pairs;

	//float* score = new float[2 * pairs];
	std::vector<float> score;
	//float* labels = new float[2 * pairs];
	std::vector<float> labels;

	for (int i = 0; i < pairs; i++){

		std::string image_name;
		std::string first_image_number, second_image_number;

		ifs >> image_name;
		ifs >> first_image_number;
		ifs >> second_image_number;

		std::string first_image_path, second_image_path;

		first_image_path = filesDirectory + image_name + "/" + image_name + "_" + string(4 - first_image_number.length(), '0') + first_image_number + ".jpg";
		second_image_path = filesDirectory + image_name + "/" + image_name + "_" + string(4 - second_image_number.length(), '0') + second_image_number + ".jpg";

		//load image
		cv::Mat first_img_color = cv::imread(first_image_path, 1);
		cv::Mat imgGray;
		cv::cvtColor(first_img_color, imgGray, CV_BGR2GRAY);

		cv::Mat second_img_color = cv::imread(second_image_path, 1);
		cv::Mat second_img_gray;
		cv::cvtColor(second_img_color, second_img_gray, CV_BGR2GRAY);

		ImageData first_img_data_color(first_img_color.cols, first_img_color.rows, first_img_color.channels());
		first_img_data_color.data = first_img_color.data;

		ImageData first_img_data_gray(imgGray.cols, imgGray.rows, imgGray.channels());
		first_img_data_gray.data = imgGray.data;

		ImageData second_img_data_color(second_img_color.cols, second_img_color.rows, second_img_color.channels());
		second_img_data_color.data = second_img_color.data;

		ImageData second_img_data_gray(second_img_gray.cols, second_img_gray.rows, second_img_gray.channels());
		second_img_data_gray.data = second_img_gray.data;

		// Detect faces
		std::vector<seeta::FaceInfo> first_faces = detector.Detect(first_img_data_gray);
		int32_t first_face_num = static_cast<int32_t>(first_faces.size());

		std::vector<seeta::FaceInfo> second_faces = detector.Detect(second_img_data_gray);
		int32_t second_face_num = static_cast<int32_t>(second_faces.size());

		if (first_face_num == 0 || second_face_num == 0)
		{
			if (first_face_num == 0)
			{
				std::cout << first_image_path << ": " << "Face are not detected." << endl;
			}
			if (second_face_num == 0)
			{
				std::cout << second_image_path << ": " << "Face are not detected." << endl;
			}
			continue;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark first_points[5];
		point_detector.PointDetectLandmarks(first_img_data_gray, first_faces[0], first_points);

		seeta::FacialLandmark second_points[5];
		point_detector.PointDetectLandmarks(second_img_data_gray, second_faces[0], second_points);

		// Extract face identity feature
		float* first_fea = new float[2048];
		float* second_fea = new float[2048];
		face_recognizer.ExtractFeatureWithCrop(first_img_data_color, first_points, first_fea);
		face_recognizer.ExtractFeatureWithCrop(second_img_data_color, second_points, second_fea);

		// Caculate similarity of two faces
		float sim = face_recognizer.CalcSimilarity(first_fea, second_fea);
		//std::cout << sim << endl;

		score.push_back(1.0f - sim); // cos distance
		labels.push_back(1.0f);
		delete first_fea;
		delete second_fea;
	}

	for (int i = 0; i < pairs; i++){

		std::string first_image_name, second_image_name;
		std::string first_image_number, second_image_number;

		ifs >> first_image_name;
		ifs >> first_image_number;
		ifs >> second_image_name;
		ifs >> second_image_number;

		std::string first_image_path, second_image_path;

		first_image_path = TRAIN_TEST_PATH + first_image_name + "/" + first_image_name + "_" + string(4 - first_image_number.length(), '0') + first_image_number + ".jpg";
		second_image_path = TRAIN_TEST_PATH + second_image_name + "/" + second_image_name + "_" + string(4 - second_image_number.length(), '0') + second_image_number + ".jpg";

		//load image
		cv::Mat first_img_color = cv::imread(first_image_path, 1);
		cv::Mat imgGray;
		cv::cvtColor(first_img_color, imgGray, CV_BGR2GRAY);

		cv::Mat second_img_color = cv::imread(second_image_path, 1);
		cv::Mat second_img_gray;
		cv::cvtColor(second_img_color, second_img_gray, CV_BGR2GRAY);

		ImageData first_img_data_color(first_img_color.cols, first_img_color.rows, first_img_color.channels());
		first_img_data_color.data = first_img_color.data;

		ImageData first_img_data_gray(imgGray.cols, imgGray.rows, imgGray.channels());
		first_img_data_gray.data = imgGray.data;

		ImageData second_img_data_color(second_img_color.cols, second_img_color.rows, second_img_color.channels());
		second_img_data_color.data = second_img_color.data;

		ImageData second_img_data_gray(second_img_gray.cols, second_img_gray.rows, second_img_gray.channels());
		second_img_data_gray.data = second_img_gray.data;

		// Detect faces
		std::vector<seeta::FaceInfo> first_faces = detector.Detect(first_img_data_gray);
		int32_t first_face_num = static_cast<int32_t>(first_faces.size());

		std::vector<seeta::FaceInfo> second_faces = detector.Detect(second_img_data_gray);
		int32_t second_face_num = static_cast<int32_t>(second_faces.size());

		if (first_face_num == 0 || second_face_num == 0)
		{
			if (first_face_num == 0)
			{
				std::cout << first_image_path << ": " << "Face are not detected." << endl;
			}
			if (second_face_num == 0)
			{
				std::cout << second_image_path << ": " << "Face are not detected." << endl;
			}
			continue;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark first_points[5];
		point_detector.PointDetectLandmarks(first_img_data_gray, first_faces[0], first_points);

		seeta::FacialLandmark second_points[5];
		point_detector.PointDetectLandmarks(second_img_data_gray, second_faces[0], second_points);

		// Extract face identity feature
		float* first_fea = new float[2048];
		float* second_fea = new float[2048];
		face_recognizer.ExtractFeatureWithCrop(first_img_data_color, first_points, first_fea);
		face_recognizer.ExtractFeatureWithCrop(second_img_data_color, second_points, second_fea);

		// Caculate similarity of two faces
		float sim = face_recognizer.CalcSimilarity(first_fea, second_fea);
		//std::cout << sim << endl;

		score.push_back(1.0f - sim); // cos distance
		labels.push_back(0.0f);
		delete first_fea;
		delete second_fea;
	}

	for (int i = 0; i < score.size(); i++){
		ofs << score[i] << " " << labels[i] << endl;
	}

	int N = 100;

	float score_max = 0.0f, score_min = 1.0f;

	for (int i = 0; i < score.size(); i++){

		if (score[i] < score_min)
			score_min = score[i];

		if (score[i] > score_max)
			score_max = score[i];
	}

	cout << score_min << " " << score_max << endl;

	float step = (score_max - score_min) / (N - 1);
	float* thresh = new float[N];
	float* accuracy = new float[N];

	for (int i = 0; i < N; i++){
		thresh[i] = score_min + step*i;
		accuracy[i] = 0;
	}

	float M = 0.0f, M_thresh = 0.0f;
	float eps = 0.001;

	for (int i = 0; i < N; i++){

		float* res = new float[score.size()];

		for (int j = 0; j < score.size(); j++){

			if (score[j] < thresh[i])
				res[j] = 1.0f;
			else
				res[j] = 0.0f;
		}

		int right_ans = 0;

		for (int j = 0; j < score.size(); j++){

			if (abs(labels[j] - res[j]) < eps)
				right_ans++;
		}

		accuracy[i] = ((float)right_ans) / (score.size());
		if (accuracy[i] > M){
			M_thresh = thresh[i];
			M = accuracy[i];
		}

		delete res;
	}

	float* res = new float[score.size()];

	for (int j = 0; j < score.size(); j++){

		if (score[j] < M_thresh_test)
			res[j] = 1.0f;
		else
			res[j] = 0.0f;
	}

	int right_ans = 0;

	for (int j = 0; j < score.size(); j++){

		if (abs(labels[j] - res[j]) < eps)
			right_ans++;
	}

	float accur = ((float)right_ans) / (score.size());

	delete res;

	cout << M << " " << M_thresh << endl;

	cout << accur << endl;

	delete thresh;
	ofs.close();
	ifs.close();
	return;

}

void getDataFromFile(){

	std::ifstream ifs;
	ifs.open("outputs.txt");

	std::vector<float> score;
	std::vector<float> labels;

	float scoreitem, labelitem;

	while (ifs >> scoreitem, ifs >> labelitem){
		score.push_back(1.0f - scoreitem);
		labels.push_back(labelitem);
	}

	float *scorearray = &score[0];
	float *labelsarray = &labels[0];

	int N = 200;

	float score_max = 0.0f, score_min = 1.0f;

	for (int i = 0; i < score.size(); i++){

		if (score[i] < score_min)
			score_min = score[i];

		if (score[i] > score_max)
			score_max = score[i];
	}

//	cout << score_min << " " << score_max << endl;

	float step = (score_max - score_min) / (N - 1);
	float* thresh = new float[N];
	float* accuracy = new float[N];

	for (int i = 0; i < N; i++){
		thresh[i] = score_min + step*i;
		accuracy[i] = 0;
	}

	float M = 0.0f, M_thresh = 0.0f;
	float eps = 0.001;

	for (int i = 0; i < N; i++){

		float* res = new float[score.size()]; 

		for (int j = 0; j < score.size(); j++){

			if (score[j] < thresh[i]){
				res[j] = 1.0f;
			}
			else
				res[j] = 0.0f;
		}

		int right_ans = 0;

		for (int j = 0; j < score.size(); j++){

			if (abs(labels[j] - res[j]) < eps)
				right_ans++;
		}

		accuracy[i] = ((float)right_ans) / (score.size());
		if (accuracy[i] > M){
			M_thresh = thresh[i];
			M = accuracy[i];
		}

		delete res;
	}

	cout << M << " " << M_thresh << endl;

}

void allFilesVectorPrint(){

	std::string pathToList = "C:/exposoft/FR_test/output_files_list.txt";
	std::string destinationPath = "C:/exposoft/FR_test/lfw_crop_faces/";
	std::string vectorsList = "C:/exposoft/FR_test/output_vector_list.txt";

	std::ifstream ifs;
	ifs.open(pathToList, std::ifstream::in);

	std::ofstream ofs;
	ofs.open(vectorsList, std::ofstream::out);

	std::string pathToFile;

	// Initialize face detection model
	seeta::FaceDetection detector("../../../FaceDetection/model/seeta_fd_frontal_v1.0.bin");
	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f);
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// Initialize face alignment model 
	seeta::FaceAlignment point_detector("../../../FaceAlignment/model/seeta_fa_v1.1.bin");

	// Initialize face Identification model 
	FaceIdentification face_recognizer((MODEL_DIR + "seeta_fr_v1.0.bin").c_str());

	while (ifs >> pathToFile){

		cv::Mat imgColor = cv::imread(pathToFile, 1);
		cv::Mat imgGray;
		cv::cvtColor(imgColor, imgGray, CV_BGR2GRAY);

		ImageData imgDataColor(imgColor.cols, imgColor.rows, imgColor.channels());
		imgDataColor.data = imgColor.data;

		ImageData imgDataGray(imgGray.cols, imgGray.rows, imgGray.channels());
		imgDataGray.data = imgGray.data;


		// Detect faces
		std::vector<seeta::FaceInfo> faces = detector.Detect(imgDataGray);
		int32_t faceNum = static_cast<int32_t>(faces.size());

		if (faceNum == 0)
		{
			std::cout << pathToFile << ": " << "Face are not detected." << endl;
			continue;
		}

		// Detect 5 facial landmarks
		seeta::FacialLandmark points[5];
		point_detector.PointDetectLandmarks(imgDataGray, faces[0], points);

		// Create a images to store crop face.
		cv::Mat cropImg(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC(face_recognizer.crop_channels()));
		ImageData cropImgData(cropImg.cols, cropImg.rows, cropImg.channels());
		cropImgData.data = cropImg.data;

		face_recognizer.CropFace(imgDataColor, points, cropImgData);

		std::string directory = pathToFile.substr(0, pathToFile.find_last_of("\\"));
		directory = directory.substr(directory.find_last_of("\\") + 1);
		std::string imageName = pathToFile.substr(pathToFile.find_last_of("\\") + 1);

		cv::imwrite(destinationPath + directory + "/" + imageName, cropImg);

		// Extract face identity feature
		float faceVector[2048];
		face_recognizer.ExtractFeatureWithCrop(imgDataColor, points, faceVector);

		ofs << imageName;
		for (int i = 0; i < 2048; i++){
			ofs << ", " << faceVector[i];
		}
		ofs << endl;

	}

}

int main(int argc, char* argv[]) {
	//trainTest(argc, argv);
	//getDataFromFile();
	//allFilesVectorPrint();
	threshTest();
}


