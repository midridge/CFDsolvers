//
// Created by ennis on 2021/5/30.
//
#include "utils.h"
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

bool exportResult(const string &name, unsigned time_step_count, Array &x, Array &y, Array &u, Array &v, Array &p) {
    string filename = name + "_";
    filename += to_string(time_step_count);
    filename += ".vtk";
    ofstream f_out(filename.c_str());

    long n_x_points = x.cols();
    long n_y_points = x.rows();
    long n_points = n_x_points * n_y_points;

    if (!f_out.is_open()) return false;

    f_out << "# vtk DataFile Version 3.0\n";
    f_out << "RESULT FILE\n";
    f_out << "ASCII\n\n";
    f_out << "DATASET STRUCTURED_GRID\n";
    f_out << "DIMENSIONS\t" << n_x_points << '\t' << n_y_points << '\t' << "1\n";
    f_out << "POINTS\t" << n_points << "\tdouble\n";
    for (long i = 0; i < n_y_points; i++) {
        for (long j = 0; j < n_x_points; j++)
            f_out << x(i, j) << '\t' << y(i, j) << "\t0.0\n";
    }

    f_out << "\nPOINT_DATA\t" << n_points << endl;
    f_out << "SCALARS\tp\tdouble\n";
    f_out << "LOOKUP_TABLE default\n";
    for (long i = 0; i < n_y_points; i++) {
        for (long j = 0; j < n_x_points; j++)
            f_out << p(i, j) << endl;
    }

    f_out << "SCALARS\tu\tdouble\n";
    f_out << "LOOKUP_TABLE default\n";
    for (long i = 0; i < n_y_points; i++) {
        for (long j = 0; j < n_x_points; j++)
            f_out << u(i, j) << endl;
    }

    f_out << "SCALARS\tv\tdouble\n";
    f_out << "LOOKUP_TABLE default\n";
    for (long i = 0; i < n_y_points; i++) {
        for (long j = 0; j < n_x_points; j++)
            f_out << v(i, j) << endl;
    }

    f_out << "SCALARS\tvelocity_magnitude\tdouble\n";
    f_out << "LOOKUP_TABLE default\n";
    for (long i = 0; i < n_y_points; i++) {
        for (long j = 0; j < n_x_points; j++)
            f_out << sqrt(u(i, j) * u(i, j) + v(i, j) * v(i, j)) << endl;
    }

    f_out << "VECTORS\tvelocity\tdouble\n";
    for (long i = 0; i < n_y_points; i++) {
        for (long j = 0; j < n_x_points; j++)
            f_out << u(i, j) << '\t' << v(i, j) << "\t0.0\n";
    }

    f_out.close();
    return true;
}

bool readResult(const std::string &name, Array &u, Array &v, Array &p) {
    if (name.find(".vtk") == string::npos) {
        cout << "Unsupported data format.\n";
        return false;
    }
    ifstream f_in(name);

    if (!f_in.is_open()) {
        cout << "Cannot open file " << name << endl;
        return false;
    }

    std::string line;
    std::stringstream sst;

    int n_x_grids, n_y_grids;
    string temp;

    while (getline(f_in, line)) {
        if (line == "DATASET STRUCTURED_GRID") {
            getline(f_in, line);
            sst << line;
            sst >> temp >> n_x_grids >> n_y_grids;
            sst.str("");
            break;
        }
    }
    if (n_x_grids != u.cols() || n_y_grids != u.rows()) {
        cout << "Grid does not match.\n";
        cout << " - File field grid number for x: " << n_x_grids << endl;
        cout << " - File field grid number for y: " << n_y_grids << endl;
        cout << " - Target field grid number for x: " << u.cols() << endl;
        cout << " - Target field grid number for y: " << u.rows() << endl;
        return false;
    }

    while (getline(f_in, line)) {
        if (line == "SCALARS\tp\tdouble") {
            getline(f_in, line);
            for (int i = 0; i < n_y_grids; i++) {
                for (int j = 0; j < n_x_grids; j++) {
                    getline(f_in, line);
                    p(i, j) = stod(line);
                }
            }
        } else if (line == "SCALARS\tu\tdouble") {
            getline(f_in, line);
            for (int i = 0; i < n_y_grids; i++) {
                for (int j = 0; j < n_x_grids; j++) {
                    getline(f_in, line);
                    u(i, j) = stod(line);
                }
            }
        } else if (line == "SCALARS\tv\tdouble") {
            getline(f_in, line);
            for (int i = 0; i < n_y_grids; i++) {
                for (int j = 0; j < n_x_grids; j++) {
                    getline(f_in, line);
                    v(i, j) = stod(line);
                }
            }
        }
    }

    f_in.close();
    return true;

}