#include <bits/stdc++.h>
#include "Dataset.h"
using namespace std;

double get_leaving_one_out_accuracy(Dataset ds, vector<int> current_features)
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

void convert_feature_map_to_feature_vector(unordered_map<int, int> done_features, vector<int> &current_features)
{
    current_features.clear();
    for (auto x : done_features)
    {
        if (x.second)
            current_features.push_back(x.first);
    }
}

void backward_elimination(Dataset ds)
{
    int total_features = ds.features[0].size();
    vector<int> current_features;
    vector<int> best_features;
    double best_overall_accuracy = 0.0;
    unordered_map<int, int> done_features;
    for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
    {
        done_features[feat_idx] = 1;
    }
    convert_feature_map_to_feature_vector(done_features, current_features);
    best_features = current_features;
    double accuracy = get_leaving_one_out_accuracy(ds, current_features);
    best_overall_accuracy = accuracy; 
    cout << "Running nearest neighbor with all the features {" << current_features[0] + 1;
    for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
    {
        cout << ", " << current_features[feat_idx] + 1;
    }
    cout << "} has accuracy " << accuracy * 100 << "%\n\n";

    for (int number_of_features = total_features; number_of_features >= 2; number_of_features--)
    {
        int worst_feat = -1;
        double best_accuracy = 0;

        for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
        {
            if (!done_features[feat_idx])
                continue;
            done_features[feat_idx] = 0;
            convert_feature_map_to_feature_vector(done_features, current_features);
            accuracy = get_leaving_one_out_accuracy(ds, current_features);
            cout << "       Using feature(s) {" << current_features[0] + 1;
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
        done_features[worst_feat] = 0;

        convert_feature_map_to_feature_vector(done_features, current_features);
        cout << "\nFeature set {" << current_features[0] + 1;
        for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
        {
            cout << ", " << current_features[feat_idx] + 1;
        }
        cout << "} was best with " << best_accuracy * 100 << " accuracy.\n\n";
        if (best_accuracy > best_overall_accuracy)
        {
            best_features = current_features;
            best_overall_accuracy = best_accuracy;
        }
    }
    cout << "Finisheshed Search! Best feature subset came out to be {" << best_features[0] + 1;
    for (int feat_idx = 1; feat_idx < (int)best_features.size(); feat_idx++)
    {
        cout << ", " << best_features[feat_idx] + 1;
    }
    cout << "} with accuracy " << best_overall_accuracy * 100 << "\n";
}

void forward_selection(Dataset ds)
{
    vector<int> current_features;
    unordered_map<int, int> done_features;
    vector<int> best_features;
    double best_overall_accuracy = 0;
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
            double accuracy = get_leaving_one_out_accuracy(ds, current_features);

            cout << "       Using feature(s) {" << current_features[0] + 1;
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
        current_features.push_back(best_feat);
        done_features[best_feat] = 1;
        cout << "\nFeature set {" << current_features[0] + 1;
        for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
        {
            cout << ", " << current_features[feat_idx] + 1;
        }
        cout << "} was best with " << best_accuracy * 100 << " accuracy.\n\n";
        if (best_accuracy > best_overall_accuracy)
        {
            best_features = current_features;
            best_overall_accuracy = best_accuracy;
        }
    }
    cout << "Finisheshed Search! Best feature subset came out to be {" << best_features[0] + 1;
    for (int feat_idx = 1; feat_idx < best_features.size(); feat_idx++)
    {
        cout << ", " << best_features[feat_idx] + 1;
    }
    cout << "} with accuracy " << best_overall_accuracy * 100 << "\n\n";
}

int main()
{
    system("pwd");
    cout << "Hello World! Welcome to my Feature Selection algorithm:\n";
    cout << "Please select a dataset according to its number: \n1) CS205_small_Data__49.txt\n2) CS205_large_Data__38.txt\n3) Gender Classification (Part 2)\n";
    int file;
    cin >> file;
    Dataset *ds;
    if (file == 1)
        ds = new Dataset("CS205_small_Data__49.txt");
    else if (file == 2)
        ds = new Dataset("CS205_large_Data__38.txt");
    else if (file == 3)
        ds = new Dataset("gender_transformed.txt");
    else
        cerr << "Wrong Selection of the Dataset!\n";
    int algorithm;
    cout << "Please select an algorithm type according to its number: \n1) Forward Selection.\n2) Backward Elimination.\n";
    cin >> algorithm;
    cout << "This dataset has " << ds->features[0].size() << " features (excluding the class attribute) and have " << ds->instances << " instances.\n\n";
    cout << "Beginning Search.\n\n";
    if (algorithm == 1)
        forward_selection(*ds);
    else if (algorithm == 2)
        backward_elimination(*ds);
    else
        cerr << "Wrong Selection of the Algorithm!\n";
}