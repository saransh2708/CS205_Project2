#include <bits/stdc++.h>
#include "Dataset.h"
using namespace std;
using namespace std::chrono;

// Function to get leaving one out accuracy with the features in current_features
double get_leaving_one_out_accuracy(Dataset ds, vector<int> current_features)
{
    int rows = ds.instances;
    int correctly_labelled = 0;

    // Iterating over every row and finding other row which is closest to this row when compared to features in current_features.
    for (int row1 = 0; row1 < rows; row1++)
    {
        int closest = -1;
        double minimum_distance = 1e12;
        for (int row2 = 0; row2 < rows; row2++)
        {
            if (row1 == row2)
                continue; // Skipping comparing the same rows
            double distance = 0;

            // Below 5 lines calculated the Euclidian Distance between 2 features.
            for (auto feat_idx : current_features)
            {
                distance += pow(ds.features[row1][feat_idx] - ds.features[row2][feat_idx], 2); 
            }
            distance = sqrt(distance);

            // Finding the closest row to the row1.
            if (distance < minimum_distance)
            {
                minimum_distance = distance;
                closest = row2;
            }
        }
        // If both the rows are same labelled then marked as correct. 
        correctly_labelled += ds.labels[row1] == ds.labels[closest];
    }
    // Return the accuracy
    return (double)correctly_labelled / rows;
}

// This is a helper function to convert feature map to vector. This is relevant to current implementation and not to the algorithm
void convert_feature_map_to_feature_vector(unordered_map<int, int> done_features, vector<int> &current_features)
{
    current_features.clear();
    for (auto x : done_features)
    {
        if (x.second)
            current_features.push_back(x.first);
    }
}

// This is to get the accuracy if we select all the features which provides a good benchmark for selecting a subset of features.
void get_all_features_accuracy(Dataset ds)
{
    int total_features = ds.features[0].size();
    vector<int> current_features;

    // Pushing every feature in current_features list
    for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
    {
        current_features.push_back(feat_idx);
    }
    double accuracy = get_leaving_one_out_accuracy(ds, current_features);

    // Logging the results
    cout << "Running nearest neighbor with all the features {" << current_features[0] + 1;
    for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
    {
        cout << ", " << current_features[feat_idx] + 1;
    }
    cout << "} has accuracy " << accuracy * 100 << "%\n\n";
}

// This function is the implementation of Backward Elimination algorithm.
void backward_elimination(Dataset ds)
{
    int total_features = ds.features[0].size();
    vector<int> current_features;
    vector<int> best_features;
    double best_overall_accuracy = 0.0;
    unordered_map<int, int> done_features;

    // Marking every feature as included
    for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
    {
        done_features[feat_idx] = 1;
    }

    // Using this helper function to update current_features vector.
    convert_feature_map_to_feature_vector(done_features, current_features);

    best_features = current_features;

    // Getting initial accuracy.
    double accuracy = get_leaving_one_out_accuracy(ds, current_features);
    best_overall_accuracy = accuracy;

    // Main loop which eliminates a feature in every iteration.
    for (int number_of_features = total_features; number_of_features >= 2; number_of_features--)
    {
        int worst_feat = -1;
        double best_accuracy = 0;

        for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
        {
            if (!done_features[feat_idx])
                continue;
            
            // Removing this feature and checking the accuracy of the current feature set.
            done_features[feat_idx] = 0;
            convert_feature_map_to_feature_vector(done_features, current_features);
            accuracy = get_leaving_one_out_accuracy(ds, current_features);

            cout << "       Using feature(s) {" << current_features[0] + 1;
            for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
            {
                cout << ", " << current_features[feat_idx] + 1;
            }
            cout << "} accuracy is " << accuracy * 100 << "%\n";

            // If there is an increase in the accuracy then store it.
            if (accuracy > best_accuracy)
            {
                best_accuracy = accuracy;
                worst_feat = feat_idx;
            }

            // Add back the feature to the current feature set.
            done_features[feat_idx] = 1;
        }

        // Fially remove the worst feature.
        done_features[worst_feat] = 0;

        // Update the feature set.
        convert_feature_map_to_feature_vector(done_features, current_features);

        // Logging which feature gave the best results. 
        cout << "\nFeature set {" << current_features[0] + 1;
        for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
        {
            cout << ", " << current_features[feat_idx] + 1;
        }
        cout << "} was best with " << best_accuracy * 100 << " accuracy.\n\n";

        // Storing the best overall accuracy and features that resulted it.
        if (best_accuracy > best_overall_accuracy)
        {
            best_features = current_features;
            best_overall_accuracy = best_accuracy;
        }
    }

    // Logging the final results.
    cout << "Finisheshed Search! Best feature subset came out to be {" << best_features[0] + 1;
    for (int feat_idx = 1; feat_idx < (int)best_features.size(); feat_idx++)
    {
        cout << ", " << best_features[feat_idx] + 1;
    }
    cout << "} with accuracy " << best_overall_accuracy * 100 << "%\n";
}

// This function is the implementation of Forward Selection algorithm.
void forward_selection(Dataset ds)
{
    int total_features = ds.features[0].size();
    vector<int> current_features;
    unordered_map<int, int> done_features;
    vector<int> best_features;
    double best_overall_accuracy = 0;
    
    // Main loop which adds a feature in every iteration.
    for (int number_of_features = 1; number_of_features <= total_features; number_of_features++)
    {
        int best_feat = -1;
        double best_accuracy = 0;

        // Iterating over every feature to see their contribution to the accuracy.
        for (int feat_idx = 0; feat_idx < total_features; feat_idx++)
        {
            if (done_features[feat_idx])
                continue;
            
            // Adding to the current feature set. 
            current_features.push_back(feat_idx);
            double accuracy = get_leaving_one_out_accuracy(ds, current_features);

            // Logging results of the above addition.
            cout << "       Using feature(s) {" << current_features[0] + 1;
            for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
            {
                cout << ", " << current_features[feat_idx] + 1;
            }
            cout << "} accuracy is " << accuracy * 100 << "%\n";

            // If there is an increase in the accuracy then store it. 
            if (accuracy > best_accuracy)
            {
                best_accuracy = accuracy;
                best_feat = feat_idx;
            }

            // Removing the current feature.
            current_features.pop_back();
        }

        // Adding the best feature to the current set.
        current_features.push_back(best_feat);
        done_features[best_feat] = 1;

        // Logging the results
        cout << "\nFeature set {" << current_features[0] + 1;
        for (int feat_idx = 1; feat_idx < current_features.size(); feat_idx++)
        {
            cout << ", " << current_features[feat_idx] + 1;
        }
        cout << "} was best with " << best_accuracy * 100 << " accuracy.\n\n";

         // Storing the best overall accuracy and features that resulted it.
        if (best_accuracy > best_overall_accuracy)
        {
            best_features = current_features;
            best_overall_accuracy = best_accuracy;
        }
    }

    // Logging the final results.
    cout << "Finisheshed Search! Best feature subset came out to be {" << best_features[0] + 1;
    for (int feat_idx = 1; feat_idx < best_features.size(); feat_idx++)
    {
        cout << ", " << best_features[feat_idx] + 1;
    }
    cout << "} with accuracy " << best_overall_accuracy * 100 << "%\n\n";
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
    
    get_all_features_accuracy(*ds);
    
    cout << "Beginning Search.\n\n";

    auto start = high_resolution_clock::now();
    if (algorithm == 1)
        forward_selection(*ds);
    else if (algorithm == 2)
        backward_elimination(*ds);
    else
        cerr << "Wrong Selection of the Algorithm!\n";
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Time taken by the algorithm: " << (double)duration.count() / 60000.0 << " minutes\n";
}