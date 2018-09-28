void normalize_max_min(vector<float>& v, float offset, float scale_coeff)
{
	int sz = v.size();
	float max_v = -FLT_MAX;
	float min_v = FLT_MAX;
	for (int i = 0; i < sz; i++)
	{
		if (v[i] > max_v) max_v = v[i];
		if (v[i] < min_v) min_v = v[i];
	}
	float scale = scale_coeff / (max_v - min_v);
	for (int i = 0; i < sz; i++)
	{
		v[i] = (v[i] - min_v) * scale + offset;
	}
}

void genDepthImage(vector<float> x, vector<float> y, vector<float> z, string fname)
{
	const int window = 4;
	const int width = 384;
	const int height = 512;
	normalize_max_min(x, 0, width);
	normalize_max_min(y, height, -height);
	normalize_max_min(z, 0, 255);
	//Mat_<Vec3f> image = Mat::zeros(height, width, CV_32FC3);
	Mat_<float> image = Mat::zeros(height, width, CV_32FC1);
	Mat_<float> num = Mat::zeros(height, width, CV_32FC1);
	for (int i = 0; i < x.size(); i++)
	{
		int u0 = int(x[i]), v0 = int(y[i]);
		for (int p = -window; p <= window; p++)
		{
			for (int q = -window; q <= window; q++)
			{
				int u = u0 + p;
				int v = v0 + q;
				if (u >= 0 && u < width && v >= 0 && v < height)
				{
					//image[v][u][0] += ((i - 1) / 50) * 255.0 / 200;
					//image[v][u][2] += ((i - 1) % 50) * 255.0 / 50;
					//image[v][u][1] += z[i];
					image[v][u] += z[i];
					num[v][u]++;
				}
			}
		}
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			//for (int k = 0; k < 3; k++)
			//{
			//	if (num[i][j]) image[i][j][k] /= num[i][j];
			//	//image[i][j][k] = 255 - image[i][j][k];
			//}
			image[i][j] /= num[i][j];
		}
	}
	imwrite(fname, image);
}