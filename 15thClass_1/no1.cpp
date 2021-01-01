#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void KMeans(void);

int main()
{
	//开始计时
	double start = static_cast<double>(getTickCount());
	KMeans();
	//结束计时
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//显示时间
	cout << "processing time:" << time / 1000 << "ms" << endl;
	//等待键盘响应，按任意键结束程序
	system("pause");
	return 0;
}

void KMeans(void)
{
	const int MAX_CLUSTERS = 5;//类的最大种数
	//颜色表
	Scalar colorTab[] =
	{
		Scalar(0, 0, 255),
		Scalar(0,255,0),
		Scalar(255,100,100),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};

	Mat img(500, 500, CV_8UC3);//特征空间
	RNG rng(12345);

	for (;;)
	{
		int k, clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);//随机类种数
		int i, sampleCount = rng.uniform(1, 1001);            //随机样本数
		Mat points(sampleCount, 1, CV_32FC2), labels;         //随机样本序列

		clusterCount = MIN(clusterCount, sampleCount);
		std::vector<Point2f> centers;

		//高斯分布生成随机样本 
		for (k = 0; k < clusterCount; k++)
		{
			Point center;
			center.x = rng.uniform(0, img.cols);
			center.y = rng.uniform(0, img.rows);
			Mat pointChunk = points.rowRange(k * sampleCount / clusterCount,
				k == clusterCount - 1 ? sampleCount :
				(k + 1) * sampleCount / clusterCount);
			rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols * 0.05, img.rows * 0.05)); //用随机数填充矩阵
		}
		//随机排列
		randShuffle(points, 1, &rng);

		double compactness = kmeans(points, clusterCount, labels,
			TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);

		//清除特征映射空间
		img = Scalar::all(0);

		//绘出映射点
		for (i = 0; i < sampleCount; i++)
		{
			int clusterIdx = labels.at<int>(i);
			Point ipt = points.at<Point2f>(i);
			circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
		}
		//绘出聚类结果
		for (i = 0; i < (int)centers.size(); ++i)
		{
			Point2f c = centers[i];
			circle(img, c, 40, colorTab[i], 1, LINE_AA);
		}
		cout << "Compactness: " << compactness << endl;
		//显示聚类结果
		imshow("clusters", img);

		char key = (char)waitKey();
		if (key == 27 || key == 'q' || key == 'Q') // 按下'ESC'退出
			break;
	}

}