/**
* This is a modified version of TemplatedVocabulary.h from DBoW2 (see below).
* Added functions: Save and Load from text files without using cv::FileStorage.
* Date: August 2015
* Raúl Mur-Artal
*/

/**
* File: TemplatedVocabulary.h
* Date: February 2011
* Author: Dorian Galvez-Lopez
* Description: templated vocabulary
* License: see the LICENSE.txt file
*
*/
#include"FORB.h"
#include"TemplatedVocabulary.h"
#include "tinydir.h"
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace DBoW2;

void get_file_names(string dir_name, vector<string> & names)
{
	names.clear();
	tinydir_dir dir;
	tinydir_open(&dir, dir_name.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (!file.is_dir)
		{
			names.push_back(file.path);
		}
		tinydir_next(&dir);
	}
	tinydir_close(&dir);
}

int main()
{
	typedef TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
	ORBVocabulary* mpVocabulary;
	mpVocabulary = new ORBVocabulary();
	
	//加载预训练词典
	bool bVocLoad = mpVocabulary->loadFromBinaryFile("C:\\Users\\sggzg\\Desktop\\DBoW2\\ORBvoc.bin");
	if (!bVocLoad)
	{
		cerr << "Wrong path to vocabulary. " << endl;
		cerr << "Falied to open at:ORBvoc.bin "<< endl;
		exit(-1);
	}
	cout << "Vocabulary loaded!" << endl << endl;

	//使用词典
	Ptr<Feature2D> fdetector;
	fdetector = ORB::create();
	vector<string> img_names;
	vector<Mat> features;
	BowVector v1, v2;
	get_file_names("C:\\Users\\sggzg\\Desktop\\DJ", img_names);

	for (int i = 0; i < img_names.size(); i++)
	{
		Mat image = cv::imread(img_names[i], 0);
		vector<KeyPoint>  keypoints;
	    Mat descriptors;
		fdetector->detectAndCompute(image, Mat(), keypoints, descriptors);
		features.push_back(descriptors);
	}

	clock_t start, finish;
	double totaltime;
	start = clock();

	for (int i = 0; i < features.size(); i++)
	{
		mpVocabulary->transform(features[1],v1);
		mpVocabulary->transform(features[i],v2);
		cout << "Image " << "2" << " vs Image " << i + 1 << ": " << mpVocabulary->score(v1,v2) << endl;
	}

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "\n此程序的运行时间为" << totaltime << "秒！" << endl;

	system("pause");
	return 0;
}
