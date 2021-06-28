#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iomanip>
#include <vector>
#include "omp.h"
#include "utils.h"
#include <chrono>
#include <ratio>
#include <string>
#include <fstream>

using namespace std;
typedef Eigen::ArrayXXd Array;
typedef Eigen::Triplet<double> Trip;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
enum { RMS, MAX };
void cal(int n_grid,
         double d_viscosity = 0.001,
         double density = 1.0,
         int s_freq = 20,
         double urf_p = 0.5,
         double urf_m = 0.7,
         int converge_metric = MAX,
         double converge = 1e-6,
         unsigned n_cores = 6,
         unsigned start_step = 0,
         bool write_log = false,
         int log_freq = 10);

double max(double x) { return (x > 0) ? x : 0; }

bool exists(const std::string& name) {
    ifstream fin(name);
    if (fin.good()) {
        fin.close();
        return true;
    } else return false;
}

int main() {
    int n_grid, s_freq, metric;
    double d_viscosity, density, urf_p, urf_m, converge;
    unsigned n_cores, start_step, log_freq;
    char write_log;
    string c_metric;
    cout << "Enter num of grids: ";
    cin >> n_grid;
    cout << "Enter dynamic viscosity: ";
    cin >> d_viscosity;
    cout << "Enter density: ";
    cin >> density;
    cout << "Enter save frequency: ";
    cin >> s_freq;
    cout << "Enter under-relaxation factor for pressure: ";
    cin >> urf_p;
    cout << "Enter under-relaxation factor for momentum: ";
    cin >> urf_m;
    cout << "Enter convergence metric [MAX/RMS]: ";
    cin >> c_metric;
    cout << "Enter convergence criteria: ";
    cin >> converge;
    cout << "Enter num of cores: ";
    cin >> n_cores;
    cout << "Enter start step: ";
    cin >> start_step;
    cout << "Write log? [y/n]: ";
    cin >> write_log;
    cin.get();

    if (c_metric.find("RMS") != string::npos)
        metric = RMS;
    else
        metric = MAX;
    if (string("yesYESYes").find(write_log) != string::npos) {
        cout << "Enter log-writing frequency: ";
        cin >> log_freq;
        cal(n_grid, d_viscosity, density, s_freq, urf_p, urf_m, metric, converge, n_cores, start_step, true, log_freq);
    } else cal(n_grid, d_viscosity, density, s_freq, urf_p, urf_m, metric, converge, n_cores, start_step, false);
//    cal(20, 0.001, 1.0, 20, 0.3, 0.6, RMS, 1e-6, 8, 2, true, 1);
//    cal(40, 0.001, 1.0, 20, 0.3, 0.6, RMS, 1e-6, 8, 2, true, 1);
//    cal(80, 0.001, 1.0, 20, 0.3, 0.6, RMS, 1e-6, 8, 2, true, 1);
//    cal(150, 0.001, 1.0, 20, 0.3, 0.6, RMS, 1e-6, 8, 2, true, 1);
//    cal(300, 0.001, 1.0, 20, 0.3, 0.6, RMS, 1e-6, 8, 2, true, 1);
    return 0;
}

void cal(int n_grid,
         double d_viscosity,
         double density,
         int s_freq,
         double urf_p,
         double urf_m,
         int converge_metric,
         double converge,
         unsigned n_cores,
         unsigned start_step,
         bool write_log,
         int log_freq) {

    auto program_begin = chrono::steady_clock::now();
    auto inner_begin = chrono::steady_clock::now();
    auto end = chrono::steady_clock::now();
    chrono::duration<double> diff = end - program_begin;
    chrono::duration<double> inner_diff = end - inner_begin;
    string save_file_name = "Re";
    save_file_name += to_string(int(1 / (d_viscosity / density)));
    save_file_name += "_Mesh";
    save_file_name += to_string(n_grid);

    string read_file_name = save_file_name + "_" + to_string(start_step) + ".vtk";

    cout << "Initializing...\n";

    unsigned num_threads = n_cores;
    omp_set_num_threads(num_threads);
    Eigen::setNbThreads(num_threads);

    /*
     * Mesh info
     */
    unsigned n_x_element = n_grid;
    unsigned n_y_element = n_grid;
    double x_range = 1;
    double y_range = 1;
    unsigned n_elements = n_y_element * n_x_element;

    /*
     * Boundary condition
     */
    double wall_velocity_n[] = { 1.0, 0.0 };
    double wall_velocity_s[] = { 0.0, 0.0 };
    double wall_velocity_e[] = { 0.0, 0.0 };
    double wall_velocity_w[] = { 0.0, 0.0 };

    /*
     * Control info
     */
    unsigned start = start_step;
    double criteria = converge;
    int save_freq = s_freq;

    /*
     * Under-relaxation factor
     */
    double p_ratio = urf_p; // pressure factor
    double m_ratio = urf_m; // momentum factor

    /*
     * Fluid properties
     */
    double rho = density;
    double mu = d_viscosity;

    /*
     * Mesh generation
     */
    Array Delta_x = x_range / n_x_element * Array::Ones(n_y_element + 2, n_x_element + 2);
    Array Delta_y = y_range / n_y_element * Array::Ones(n_y_element + 2, n_x_element + 2);
    Delta_x.col(0) = Delta_x.col(Delta_x.cols() - 1) = 0;
    Delta_y.row(0) = Delta_y.row(Delta_y.rows() - 1) = 0;

    Array Delta_x_comp = Delta_x.block(1, 1, n_y_element, n_x_element);
    Array Delta_y_comp = Delta_y.block(1, 1, n_y_element, n_x_element);

    Array delta_x_plus = x_range / n_x_element / 2.0 * Array::Ones(n_y_element, n_x_element + 1);
    Array delta_x_minus = x_range / n_x_element / 2.0 * Array::Ones(n_y_element, n_x_element + 1);
    delta_x_plus.col(delta_x_plus.cols() - 1) = delta_x_minus.col(0) = 0;

    Array delta_y_plus = y_range / n_y_element / 2.0 * Array::Ones(n_y_element + 1, n_x_element);
    Array delta_y_minus = y_range / n_y_element / 2.0 * Array::Ones(n_y_element + 1, n_x_element);
    delta_y_plus.row(delta_y_plus.rows() - 1) = delta_y_minus.row(0) = 0;

    Array delta_x = delta_x_minus + delta_x_plus;
    Array delta_y = delta_y_minus + delta_y_plus;

    // Mesh grids
    Array x = Array::Zero(n_y_element + 2, n_x_element + 2);
    Array y = Array::Zero(n_y_element + 2, n_x_element + 2);

    x.col(0) = 0;
    x.col(x.cols() - 1) = x_range;
    y.row(0) = 0;
    y.row(y.rows() - 1) = y_range;

    for (int i = 1; i < n_y_element + 1; i++) {
        for (int j = 1; j < n_x_element + 1; j++)
            x(i, j) = x(i, j - 1) + delta_x(i - 1, j - 1);
    }

    x.row(0) = x.row(1);
    x.row(x.rows() - 1) = x.row(x.rows() - 2);

    for (int i = 1; i < n_y_element + 1; i++) {
        for (int j = 1; j < n_x_element + 1; j++)
            y(i, j) = y(i - 1, j) + delta_y(i - 1, j - 1);
    }
    y.col(0) = y.col(1);
    y.col(y.cols() - 1) = y.col(y.cols() - 2);

    /*
     * Initial field
     */
    Array u = Array::Zero(n_y_element + 2, n_x_element + 2);
    Array v = Array::Zero(n_y_element + 2, n_x_element + 2);
    Array p = Array::Zero(n_y_element + 2, n_x_element + 2);

    if (start == 0) {
        u.row(0) = wall_velocity_s[0];
        v.row(0) = wall_velocity_s[1];
        u.row(u.rows() - 1) = wall_velocity_n[0];
        v.row(v.rows() - 1) = wall_velocity_n[1];
        u.col(0) = wall_velocity_w[0];
        v.col(0) = wall_velocity_w[1];
        u.col(u.cols() - 1) = wall_velocity_e[0];
        v.col(v.cols() - 1) = wall_velocity_e[1];

        exportResult(save_file_name, 0, x, y, u, v, p);
    } else {
        if (!readResult(read_file_name, u, v, p)) {
            cout << "Error: Designated step file does not exit.\n";
            cout << " - Start from zero field.\n\n";
            exportResult(save_file_name, 0, x, y, u, v, p);
            start = 0;
        }
    }

    /*
     * Create log file
     */
    int write_freq = log_freq;
    string log_file_name = "log_" + save_file_name + "_0" + ".txt";
    if (write_log) {
        int repeat_idx = 0;
        int repeat_digits;
        while (exists(log_file_name)) {
            repeat_digits = (int) to_string(repeat_idx).length();
            repeat_idx++;
            log_file_name.erase(log_file_name.end() - 4 - repeat_digits, log_file_name.end());
            log_file_name += to_string(repeat_idx) + ".txt";
        }

        ofstream log_out(log_file_name);
        if (!log_out.good())
            cout << "Failure: log file creation error.\n";
        else
            log_out << "LOG FOR " << save_file_name << "\n\n";
        log_out.close();
    }

    /*
     * Temporary data container
     */
    double a_E, a_W, a_N, a_S, a_P;
    double p_w, p_e, p_n, p_s;
    double u_e, u_w;
    double v_n, v_s;
    double ap_e, ap_w, ap_n, ap_s;
    Array a_Ps = Array::Zero(n_y_element + 2, n_x_element + 2);

    SpMat A(n_elements, n_elements);
    SpMat A_prime(n_elements, n_elements);

    Eigen::VectorXd b_u(n_elements);
    Eigen::VectorXd b_v(n_elements);
    Eigen::VectorXd b_p(n_elements);
    Eigen::VectorXd u_comp, v_comp;
    Eigen::VectorXd p_prime_comp;

    Array p_prime = Array::Zero(n_y_element + 2, n_x_element + 2);
    double p_prime_e, p_prime_w, p_prime_n, p_prime_s;
    Array u_prime = Array::Zero(n_y_element + 2, n_x_element + 2);
    Array v_prime = Array::Zero(n_y_element + 2, n_x_element + 2);

    double u_error, v_error, p_error;
    long u_iter, v_iter, p_iter;

    double qm;
    double b_p_norm;
    double b_p_max;
    double b_p_sum;
    Array u_error_temp;
    Array v_error_temp;
    double relative_u_error_max;
    double relative_u_error_rms;
    double relative_v_error_max;
    double relative_v_error_rms;
    bool if_converge_norm;
    bool if_converge_max;
    bool if_converge_sum;
    bool if_converge_u;
    bool if_converge_v;

    u.row(0) = wall_velocity_s[0];
    v.row(0) = wall_velocity_s[1];
    u.row(u.rows() - 1) = wall_velocity_n[0];
    v.row(v.rows() - 1) = wall_velocity_n[1];
    u.col(0) = wall_velocity_w[0];
    v.col(0) = wall_velocity_w[1];
    u.col(u.cols() - 1) = wall_velocity_e[0];
    v.col(v.cols() - 1) = wall_velocity_e[1];

    Array u_temp = u;
    Array v_temp = v;
    Array p_temp = p;

    if_converge_norm = false;
    if_converge_max = false;
    if_converge_sum = false;

    unsigned step_count = 1;
    unsigned inner_count = 0;

    cout << "Starting time loop\n\n";
    while (!if_converge_norm || !if_converge_max || !if_converge_sum || !if_converge_u || !if_converge_v) {
        inner_begin = chrono::steady_clock::now();
        cout << "Outer Iteration: " << "\t\t" << step_count << endl;

        std::vector<Trip> coefficients;
        unsigned idx = 0;
        inner_count++;
        /*
         * Get coefficients for u, v momentum equations
         */
        for (unsigned i = 0; i < n_y_element; i++) {
            for (unsigned j = 0; j < n_x_element; j++) {
                p_e = p_temp(i + 1, j + 1) * delta_x_plus(i, j + 1) / delta_x(i, j + 1) +
                      p_temp(i + 1, j + 2) * delta_x_minus(i, j + 1) / delta_x(i, j + 1);
                p_w = p_temp(i + 1, j) * delta_x_plus(i, j) / delta_x(i, j) +
                      p_temp(i + 1, j + 1) * delta_x_minus(i, j) / delta_x(i, j);
                p_n = p_temp(i + 1, j + 1) * delta_y_plus(i + 1, j) / delta_y(i + 1, j) +
                      p_temp(i + 2, j + 1) * delta_y_minus(i + 1, j) / delta_y(i + 1, j);
                p_s = p_temp(i, j + 1) * delta_y_plus(i, j) / delta_y(i, j) +
                      p_temp(i + 1, j + 1) * delta_y_minus(i, j) / delta_y(i, j);

                u_e = u_temp(i + 1, j + 1) * delta_x_plus(i, j + 1) / delta_x(i, j + 1) +
                      u_temp(i + 1, j + 2) * delta_x_minus(i, j + 1) / delta_x(i, j + 1);
                u_w = u_temp(i + 1, j) * delta_x_plus(i, j) / delta_x(i, j) +
                      u_temp(i + 1, j + 1) * delta_x_minus(i, j) / delta_x(i, j);
                v_n = v_temp(i + 1, j + 1) * delta_y_plus(i + 1, j) / delta_y(i + 1, j) +
                      v_temp(i + 2, j + 1) * delta_y_minus(i + 1, j) / delta_y(i + 1, j);
                v_s = v_temp(i, j + 1) * delta_y_plus(i, j) / delta_y(i, j) +
                      v_temp(i + 1, j + 1) * delta_y_minus(i, j) / delta_y(i, j);

                a_E = mu / delta_x(i, j + 1) * Delta_y_comp(i, j) + max(-rho * u_e * Delta_y_comp(i, j));
                a_W = mu / delta_x(i, j) * Delta_y_comp(i, j) + max(rho * u_w * Delta_y_comp(i, j));
                a_N = mu / delta_y(i + 1, j) * Delta_x_comp(i, j) + max(-rho * v_n * Delta_x_comp(i, j));
                a_S = mu / delta_y(i, j) * Delta_x_comp(i, j) + max(rho * v_s * Delta_x_comp(i, j));
                a_P = a_E + a_W + a_N + a_S;
                a_Ps(i + 1, j + 1) = a_P / m_ratio;

                b_u(idx) = Delta_y_comp(i, j) * (p_w - p_e) + (1 - m_ratio) * a_P / m_ratio * u(i + 1, j + 1);
                b_v(idx) = Delta_x_comp(i, j) * (p_s - p_n) + (1 - m_ratio) * a_P / m_ratio * v(i + 1, j + 1);

                coefficients.emplace_back(Trip(idx, idx, a_P / m_ratio));
                if ((idx + 1) % n_x_element)
                    coefficients.emplace_back(Trip(idx, idx + 1, -a_E));
                else {
                    b_u(idx) += u_temp(i + 1, j + 2) * a_E;
                    b_v(idx) += v_temp(i + 1, j + 2) * a_E;
                }

                if (idx % n_x_element)
                    coefficients.emplace_back(Trip(idx, idx - 1, -a_W));
                else {
                    b_u(idx) += u_temp(i + 1, j) * a_W;
                    b_v(idx) += v_temp(i + 1, j) * a_W;
                }

                if (idx < n_elements - n_x_element)
                    coefficients.emplace_back(Trip(idx, idx + n_x_element, -a_N));
                else {
                    b_u(idx) += u_temp(i + 2, j + 1) * a_N;
                    b_v(idx) += v_temp(i + 2, j + 1) * a_N;
                }

                if (idx >= n_x_element)
                    coefficients.emplace_back(Trip(idx, idx - n_x_element, -a_S));
                else {
                    b_u(idx) += u_temp(i, j + 1) * a_S;
                    b_v(idx) += v_temp(i, j + 1) * a_S;
                }
                idx++;
            }
        }
        A.setFromTriplets(coefficients.begin(), coefficients.end());
        a_Ps.col(0) = a_Ps.col(1);
        a_Ps.row(0) = a_Ps.row(1);
        a_Ps.col(a_Ps.cols() - 1) = a_Ps.col(a_Ps.cols() - 2);
        a_Ps.row(a_Ps.rows() - 1) = a_Ps.row(a_Ps.rows() - 2);

        /*
         * Solve for u*, v*
         */
        Eigen::BiCGSTAB<SpMat> BCGSTAB;
        BCGSTAB.setTolerance(1e-6);
        BCGSTAB.compute(A);

        u_comp = BCGSTAB.solve(b_u);
        u_error = BCGSTAB.error();
        u_iter = BCGSTAB.iterations();

        v_comp = BCGSTAB.solve(b_v);
        v_error = BCGSTAB.error();
        v_iter = BCGSTAB.iterations();

        /*
         * Update u, v after solving
         */
        idx = 0;
        for (unsigned i = 0; i < n_y_element; i++) {
            for (unsigned j = 0; j < n_x_element; j++) {
                u_temp(i + 1, j + 1) = u_comp(idx);
                v_temp(i + 1, j + 1) = v_comp(idx);
                idx++;
            }
        }

        /*
         * Get coefficients for pressure correction equation
         */
        coefficients.clear();
        idx = 0;
        for (unsigned i = 0; i < n_y_element; i++) {
            for (unsigned j = 0; j < n_x_element; j++) {
                if (idx == 0) { // Pressure reference
                    a_P = 1;
                    coefficients.emplace_back(Trip(idx, idx, a_P));
                    coefficients.emplace_back(Trip(idx, idx + 1, 0)); // a_E
                    coefficients.emplace_back(Trip(idx, idx + n_x_element, 0)); // a_N
                    b_p(idx) = 0;
                } else {
                    u_e = u_temp(i + 1, j + 1) * delta_x_plus(i, j + 1) / delta_x(i, j + 1) +
                          u_temp(i + 1, j + 2) * delta_x_minus(i, j + 1) / delta_x(i, j + 1);
                    u_w = u_temp(i + 1, j) * delta_x_plus(i, j) / delta_x(i, j) +
                          u_temp(i + 1, j + 1) * delta_x_minus(i, j) / delta_x(i, j);
                    v_n = v_temp(i + 1, j + 1) * delta_y_plus(i + 1, j) / delta_y(i + 1, j) +
                          v_temp(i + 2, j + 1) * delta_y_minus(i + 1, j) / delta_y(i + 1, j);
                    v_s = v_temp(i, j + 1) * delta_y_plus(i, j) / delta_y(i, j) +
                          v_temp(i + 1, j + 1) * delta_y_minus(i, j) / delta_y(i, j);

                    // 判断是否在边界上，如果是，则不执行修正
                    if (j != n_y_element - 1) { // 若不在e边界
                        ap_e = a_Ps(i + 1, j + 1) * delta_x_plus(i, j + 1) / delta_x(i, j + 1) +
                               a_Ps(i + 1, j + 2) * delta_x_minus(i, j + 1) / delta_x(i, j + 1);
                        u_e -= Delta_y_comp(i, j) / ap_e * (p_temp(i + 1, j + 2) - p_temp(i + 1, j + 1));
                        a_E = rho * Delta_y_comp(i, j) / ap_e * Delta_y_comp(i, j);
                        coefficients.emplace_back(Trip(idx, idx + 1, -a_E));
                    } else a_E = 0;

                    if (j != 0) { // 若不在w边界
                        ap_w = a_Ps(i + 1, j) * delta_x_plus(i, j) / delta_x(i, j) +
                               a_Ps(i + 1, j + 1) * delta_x_minus(i, j) / delta_x(i, j);
                        u_w -= Delta_y_comp(i, j) / ap_w * (p_temp(i + 1, j + 1) - p_temp(i + 1, j));
                        a_W = rho * Delta_y_comp(i, j) / ap_w * Delta_y_comp(i, j);
                        coefficients.emplace_back(Trip(idx, idx - 1, -a_W));
                    } else a_W = 0;

                    if (i != n_y_element - 1) { // 若不在n边界
                        ap_n = a_Ps(i + 1, j + 1) * delta_y_plus(i + 1, j) / delta_y(i + 1, j) +
                               a_Ps(i + 2, j + 1) * delta_y_minus(i + 1, j) / delta_y(i + 1, j);
                        v_n -= Delta_x_comp(i, j) / ap_n * (p_temp(i + 2, j + 1) - p_temp(i + 1, j + 1));
                        a_N = rho * Delta_x_comp(i, j) / ap_n * Delta_x_comp(i, j);
                        coefficients.emplace_back(Trip(idx, idx + n_x_element, -a_N));
                    } else a_N = 0;

                    if (i != 0) { // 若不在s边界
                        ap_s = a_Ps(i, j + 1) * delta_y_plus(i, j) / delta_y(i, j) +
                               a_Ps(i + 1, j + 1) * delta_y_minus(i, j) / delta_y(i, j);
                        v_s -= Delta_x_comp(i, j) / ap_s * (p_temp(i + 1, j + 1) - p_temp(i, j + 1));
                        a_S = rho * Delta_x_comp(i, j) / ap_s * Delta_x_comp(i, j);
                        coefficients.emplace_back(Trip(idx, idx - n_x_element, -a_S));
                    } else a_S = 0;

                    a_P = a_E + a_W + a_N + a_S;
                    coefficients.emplace_back(Trip(idx, idx, a_P));

                    b_p(idx) = rho * (Delta_y_comp(i, j) * (u_w - u_e) + Delta_x_comp(i, j) * (v_s - v_n));
                }
                idx++;
            }
        }
        A_prime.setFromTriplets(coefficients.begin(), coefficients.end());

        b_p_norm = b_p.norm();
        b_p_max = b_p.maxCoeff();
        b_p_sum = b_p.cwiseAbs().sum();

        /*
         * Solve for p'
         */
        BCGSTAB.setTolerance(1e-5);
        BCGSTAB.compute(A_prime);
        p_prime_comp = BCGSTAB.solve(b_p);
        p_iter = BCGSTAB.iterations();
        p_error = BCGSTAB.error();

        // Update p_prime
        idx = 0;
        for (unsigned i = 0; i < n_y_element; i++) {
            for (unsigned j = 0; j < n_x_element; j++)
                p_prime(i + 1, j + 1) = p_prime_comp(idx++);
        }

        // Linear interpolation for p' on boundaries
        for (int i = 1; i < n_y_element + 1; i++) {
            p_prime(i, 0) = p_prime(i, 1) + delta_x(i - 1, 0) / delta_x(i - 1, 1) * (p_prime(i, 1) - p_prime(i, 2));
            p_prime(i, p_prime.cols() - 1) = p_prime(i, p_prime.cols() - 2) +
                                             delta_x(i - 1, delta_x.cols() - 1) / delta_x(i - 1, delta_x.cols() - 2) *
                                             (p_prime(i, p_prime.cols() - 2) - p_prime(i, p_prime.cols() - 3));
        }

        for (int j = 1; j < n_x_element + 1; j++) {
            p_prime(0, j) = p_prime(1, j) + delta_y(0, j - 1) / delta_y(1, j - 1) * (p_prime(1, j) - p_prime(2, j));
            p_prime(p_prime.rows() - 1, j) = p_prime(p_prime.rows() - 2, j) +
                                             delta_y(delta_y.rows() - 1, j - 1) / delta_y(delta_y.rows() - 2, j - 1) *
                                             (p_prime(p_prime.rows() - 2, j) - p_prime(p_prime.rows() - 3, j));
        }
        // p' corner interpolation
        p_prime(0, 0) = (p_prime(0, 1) * delta_y(0, 0) + p_prime(1, 0) * delta_x(0, 0)) / (delta_y(0, 0) + delta_x(0, 0));
        p_prime(p_prime.rows() - 1, p_prime.cols() - 1) = (p_prime(p_prime.rows() - 1, p_prime.cols() - 2) * delta_y(delta_y.rows() - 1, delta_y.cols() - 1) +
                                                           p_prime(p_prime.rows() - 2, p_prime.cols() - 1) * delta_x(delta_x.rows() - 1, delta_x.cols() - 1))
                                                          / (delta_y(delta_y.rows() - 1, delta_y.cols() - 1) + delta_x(delta_x.rows() - 1, delta_x.cols() - 1));
        p_prime(p_prime.rows() - 1, 0) = (p_prime(p_prime.rows() - 1, 1) * delta_y(delta_y.rows() - 1, 0)
                                          + p_prime(p_prime.rows() - 2, 0) * delta_x(delta_x.rows() - 1, 0))
                                         / (delta_y(delta_y.rows() - 1, 0) + delta_x(delta_x.rows() - 1, 0));
        p_prime(0, p_prime.cols() - 1) = (p_prime(0, p_prime.cols() - 2) * delta_y(0, delta_y.cols() - 1)
                                          + p_prime(1, p_prime.cols() - 1) * delta_x(0, delta_x.cols() - 1))
                                         / (delta_y(0, delta_y.cols() - 1) + delta_x(0, delta_x.cols() - 1));

        /*
         * Compute u', v' on nodes
         */
        for (unsigned i = 0; i < n_y_element; i++) {
            for (unsigned j = 0; j < n_x_element; j++) {
                // Interpolate p' on boundaries
                p_prime_e = p_prime(i + 1, j + 1) * delta_x_plus(i, j + 1) / delta_x(i, j + 1) +
                            p_prime(i + 1, j + 2) * delta_x_minus(i, j + 1) / delta_x(i, j + 1);
                p_prime_w = p_prime(i + 1, j) * delta_x_plus(i, j) / delta_x(i, j) +
                            p_prime(i + 1, j + 1) * delta_x_minus(i, j) / delta_x(i, j);
                p_prime_n = p_prime(i + 1, j + 1) * delta_y_plus(i + 1, j) / delta_y(i + 1, j) +
                            p_prime(i + 2, j + 1) * delta_y_minus(i + 1, j) / delta_y(i + 1, j);
                p_prime_s = p_prime(i, j + 1) * delta_y_plus(i, j) / delta_y(i, j) +
                            p_prime(i + 1, j + 1) * delta_y_minus(i, j) / delta_y(i, j);

                u_prime(i + 1, j + 1) = Delta_y_comp(i, j) / a_Ps(i + 1, j + 1) * (p_prime_w - p_prime_e);
                v_prime(i + 1, j + 1) = Delta_x_comp(i, j) / a_Ps(i + 1, j + 1) * (p_prime_s - p_prime_n);
            }
        }

        /*
         * Update u, v, p
         */
        p_temp += p_ratio * p_prime;
        u_temp += u_prime;
        v_temp += v_prime;

        /*
         * Calculate mass flow on a cross section
         */
        qm = 0;
        for (int i = 0; i < n_y_element; i++)
            qm += Delta_y_comp(i, n_x_element / 2) * rho * abs(u_temp(i + 1, n_x_element / 2 + 1));


        u_error_temp = (u_temp.block(1, 1, n_y_element, n_x_element) -
                            u.block(1, 1, n_y_element, n_x_element)).abs() /
                            u_temp.block(1, 1, n_y_element, n_x_element).abs();
        v_error_temp = (v_temp.block(1, 1, n_y_element, n_x_element) -
                            v.block(1, 1, n_y_element, n_x_element)).abs().array() /
                            v_temp.block(1, 1, n_y_element, n_x_element).abs();

        relative_u_error_max = u_error_temp.maxCoeff();
        relative_v_error_max = v_error_temp.maxCoeff();

        relative_u_error_rms = sqrt((u_error_temp * u_error_temp).sum() / n_elements);
        relative_v_error_rms = sqrt((v_error_temp * v_error_temp).sum() / n_elements);

        if_converge_norm = b_p_norm / qm <= criteria;
        if_converge_max = b_p_max / qm <= criteria;
        if_converge_sum = b_p_sum / qm <= criteria;
        if (converge_metric == MAX) {
            if_converge_u = relative_u_error_max <= criteria;
            if_converge_v = relative_v_error_max <= criteria;
        } else {
            if_converge_u = relative_u_error_rms <= criteria;
            if_converge_v = relative_v_error_rms <= criteria;
        }

        end = chrono::steady_clock::now();
        diff = end - program_begin;
        inner_diff = end - inner_begin;
        cout << "Execution time:" << setw(10) << diff.count() << "s" << setw(20) << "Clock time:" << setw(10) << inner_diff.count() << "s" << "\n\n";
        cout << setw(22) << "Inner iters" << setw(23) << "Inner error\n";
        cout << setw(4) << "u:" << setw(18) << u_iter << setw(22) << u_error << endl;
        cout << setw(4) << "v:" << setw(18) << v_iter << setw(22) << v_error << endl;
        cout << setw(4) << "p:" << setw(18) << p_iter << setw(22) << p_error << endl;
        cout << "\n" << setw(27) << "Continuity error" << setw(10) << "Norm: " << setw(13) << b_p_norm / qm
                                                                    << setw(10) << "Max: " << setw(13) << b_p_max / qm
                                                                    << setw(10) << "Sum: " << setw(13) << b_p_sum / qm;
        cout << "\n" << setw(27) << "Relative velocity error" << setw(10) << "U RMS: " << setw(13) << relative_u_error_rms
                                                                 << setw(10) << "V RMS: " << setw(13) << relative_v_error_rms;
        cout << "\n" << setw(37) << "U MAX: " << setw(13) << relative_u_error_max
                     << setw(10) << "V MAX: " << setw(13) << relative_v_error_max << endl;
        for (int i = 0; i < 96; i++)
            cout << '-';
        cout << endl;

        if (write_log && step_count % write_freq == 0) {
            ofstream lout(log_file_name, ofstream::app);
            if (lout.is_open()) {
                lout << "Outer Iteration: " << "\t\t" << step_count << endl;
                lout << "Execution time:" << setw(10) << diff.count() << "s" << setw(20) << "Clock time:" << setw(10) << inner_diff.count() << "s" << "\n\n";
                lout << setw(22) << "Inner iters" << setw(23) << "Inner error\n";
                lout << setw(4) << "u:" << setw(18) << u_iter << setw(22) << u_error << endl;
                lout << setw(4) << "v:" << setw(18) << v_iter << setw(22) << v_error << endl;
                lout << setw(4) << "p:" << setw(18) << p_iter << setw(22) << p_error << endl;
                lout << "\n" << setw(27) << "Time step continuity error" << setw(10) << "Norm: " << setw(13) << b_p_norm / qm
                     << setw(10) << "Max: " << setw(13) << b_p_max / qm
                     << setw(10) << "Sum: " << setw(13) << b_p_sum / qm;
                lout << "\n" << setw(27) << "Relative velocity error" << setw(10) << "U RMS: " << setw(13) << relative_u_error_rms
                                                                         << setw(10) << "V RMS: " << setw(13) << relative_v_error_rms;
                lout << "\n" << setw(37) << "U MAX: " << setw(13) << relative_u_error_max
                             << setw(10) << "V MAX: " << setw(13) << relative_v_error_max << endl;
                for (int i = 0; i < 96; i++)
                    lout << '-';
                lout << endl;
            }
        }

        p = p_temp;
        u = u_temp;
        v = v_temp;

        /*
         * Save and add time increment
         */
        if (step_count % save_freq == 0)
            exportResult(save_file_name, step_count / save_freq + start, x, y, u, v, p);

        step_count++;
    }
    cout << "Converged.\n";
    cout << "Saving: " << step_count / save_freq + 1 + start << "\n";
    exportResult(save_file_name, step_count / save_freq + start + 1, x, y, u, v, p);
    cout << "Max u: " << u.block(1, 1, n_y_element, n_x_element).maxCoeff() << '\n';
    cout << "Max v: " << v.block(1, 1, n_y_element, n_x_element).maxCoeff() << '\n';
    cout << "Min p: " << p.block(1, 1, n_y_element, n_x_element).minCoeff() << '\n';
    cout << "Max p: " << p.block(1, 1, n_y_element, n_x_element).maxCoeff() << '\n';
}
