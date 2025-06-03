#include <bits/stdc++.h>
using namespace std;

vector<double> getTokens(string s)
{
    string res = "";
    vector<double> tokens;
    for (auto x : s)
    {
        if (x == ' ')
        {
            if (res != "")
                tokens.push_back(stod(res));
            res = "";
            continue;
        }
        res += x;
    }
    if (res != "")
        tokens.push_back(stod(res));

    return tokens;
}

void normalizeFeatures(vector<vector<double>> &features)
{
    int rows = features.size();
    int cols = features[0].size();

    vector<double> mean(cols, 0.0);
    vector<double> std(cols, 0.0);

    for (int col = 0; col < cols; col++)
    {
        // Getting mean
        for (int row = 0; row < rows; row++)
        {
            mean[col] += features[row][col];
        }
        mean[col] /= rows;
        // Getting standard deviation
        for (int row = 0; row < rows; row++)
        {
            std[col] += pow(features[row][col] - mean[col], 2);
        }
        std[col] = sqrt(std[col] / rows);
    }

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            // Avoiding divide by zero error
            if (!std[col])
            {
                features[row][col] = 0;
                continue;
            }
            // Normalized features
            features[row][col] = (features[row][col] - mean[col]) / std[col];
        }
    }
}

class Dataset
{
public:
    int instances;
    vector<vector<double>> features;
    vector<double> labels;

    Dataset(string file)
    {
        labels.clear();
        features.clear();
        ifstream f(file);

        if (!f.is_open())
        {
            cerr << "Error opening the file\n";
            return;
        }
        else
        {
            string s;
            while (getline(f, s))
            {
                vector<double> tokens = getTokens(s);

                vector<double> feat;
                labels.push_back(tokens[0]);
                for (int i = 1; i < tokens.size(); i++)
                {
                    feat.push_back(tokens[i]);
                }
                features.push_back(feat);
            }
            f.close();
            normalizeFeatures(features);
            instances = labels.size();
        }
    }
};
