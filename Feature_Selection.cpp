#include <bits/stdc++.h>
#include "Feture_Selection.h"
using namespace std;

double get_cross_validation_accuracy(Dataset ds, vector<int> current_features)
{
    int rows = ds.instances;
    int correctly_labelled = 0;
    for (int row1 = 0; row1 < rows; row1++)
    {
        int closest = -1;
        double minimum_distance = 1e12;
        for (int row2 = 0; row2 < rows; row2++)
        {
            if (row1 == row2)
                continue;
            double distance = 0;
            for (auto feat_idx : current_features)
            {
                distance += pow(ds.features[row1][feat_idx] - ds.features[row2][feat_idx], 2);
            }
            distance = sqrt(distance);
            if (distance < minimum_distance)
            {
                minimum_distance = distance;
                closest = row2;
            }
        }
        correctly_labelled += ds.labels[row1] == ds.labels[closest];
    }
    return (double)correctly_labelled / rows;
}

void convert_map_features_to_vector(unordered_map<int, int> done_features, vector<int> &current_features)
{
    current_features.clear();
    for (auto x : done_features)
    {
        if (x.second)
            current_features.push_back(x.first);
    }
}

void backward_pass(Dataset ds)
{
    int total_features = ds.features[0].size();
    vector<int> current_features;
    unordered_map<int, int> done_features;
    for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
    {
        done_features[feat_idx] = 1;
    }
    double accuracy = get_cross_validation_accuracy(ds, current_features);
    convert_map_features_to_vector(done_features, current_features);
    cout << "Using feature(s) {" << current_features[0] + 1;
    for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
    {
        cout << ", " << current_features[feat_idx] + 1;
    }
    cout << "} accuracy is " << accuracy * 100 << "%\n";

    for (int number_of_features = total_features; number_of_features >= 1; number_of_features--)
    {
        int worst_feat = -1;
        double best_accuracy = 0;

        for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
        {
            if (!done_features[feat_idx])
                continue;
            done_features[feat_idx] = 0;
            convert_map_features_to_vector(done_features, current_features);
            accuracy = get_cross_validation_accuracy(ds, current_features);
            cout << "Using feature(s) {" << current_features[0] + 1;
            for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
            {
                cout << ", " << current_features[feat_idx] + 1;
            }
            cout << "} accuracy is " << accuracy * 100 << "%\n";
            if (accuracy > best_accuracy)
            {
                best_accuracy = accuracy;
                worst_feat = feat_idx;
            }
            done_features[feat_idx] = 1;
        }
        if (worst_feat == -1)
        {
            cerr << "No further improvements\n";
        }
        done_features[worst_feat] = 0;
        convert_map_features_to_vector(done_features, current_features);
    }
}

void forward_pass(Dataset ds)
{
    vector<int> current_features;
    unordered_map<int, int> done_features;
    int total_features = ds.features[0].size();
    for (int number_of_features = 1; number_of_features <= total_features; number_of_features++)
    {
        int best_feat = -1;
        double best_accuracy = 0;
        for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
        {
            if (done_features[feat_idx])
                continue;
            current_features.push_back(feat_idx);
            double accuracy = get_cross_validation_accuracy(ds, current_features);

            cout << "Using feature(s) {" << current_features[0] + 1;
            for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
            {
                cout << ", " << current_features[feat_idx] + 1;
            }
            cout << "} accuracy is " << accuracy * 100 << "%\n";

            if (accuracy > best_accuracy)
            {
                best_accuracy = accuracy;
                best_feat = feat_idx;
            }

            current_features.pop_back();
        }
        if (best_feat == -1)
        {
            cerr << "No further improvements\n";
        }
        current_features.push_back(best_feat);
        done_features[best_feat] = 1;
        // cout<<"Feature set {}"
    }
}

int main()
{
    system("pwd");
    Dataset m("CS205_small_Data__49.txt");
    forward_pass(m);
}