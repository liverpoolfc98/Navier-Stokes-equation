#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <functional>
#include <utility>
#include <vector>
#include <fstream>

template<class Func>
class Fourier {
public:
    Fourier(int nx, int ny, Func func) : nx_(nx), ny_(ny), eigen_value_(func) {}
    double Basis(int i, int s, double h) {
        return sqrt(2) * sin(M_PI * i * s * h);
    }
    double FToD(const std::vector<double>& values, int i, int k) {
        double result = 0, h = 1. / ny_;
        for (int j = 1; j < ny_; ++j) result += Basis(j, k, h) * values[(ny_ + 1) * i + j] * h;
        return result;
    }
    std::vector<double> FToDMatrix(const std::vector<double>& values) {
        std::vector<double> result((nx_ + 1) * (ny_ + 1));
        for (size_t i = 1; i < nx_; ++i) {
            for (size_t k = 1; k < ny_; ++k) result[i * (ny_ + 1) + k] = FToD(values, i, k);
        }
        return result;
    }
    double DToC(const std::vector<double>& d, int s, int k) {
        double result = 0, h = 1. / nx_;
        for (int i = 1; i < nx_; ++i) result += Basis(i, s, h) * d[i * (ny_ + 1) + k] * h;
        return result;
    }
    std::vector<double> DToCMatrix(const std::vector<double>& d) {
        std::vector<double> result((nx_ + 1) * (ny_ + 1));
        for (size_t s = 1; s < nx_; ++s) {
            for (size_t k = 1; k < ny_; ++k)
                result[s * (ny_ + 1) + k] = DToC(d, s, k) / eigen_value_(s, k);
        }
        return result;
    }
    double CToD(const std::vector<double>& c, int i, int k) {
        double result = 0, h = 1. / nx_;
        for (size_t s = 1; s < nx_; ++s) result += c[s * (ny_ + 1) + k] * Basis(i, s, h);
        return result;
    }
    std::vector<double> CToDMatrix(const std::vector<double>& c) {
        std::vector<double> result((nx_ + 1) * (ny_ + 1));
        for (size_t i = 1; i < nx_; ++i) {
            for (size_t k = 1; k < ny_; ++k) result[i * (ny_ + 1) + k] = CToD(c, i, k);
        }
        return result;
    }
    double DToF(const std::vector<double>& d, int i, int j) {
        double result = 0, h = 1. / ny_;
        for (size_t k = 1; k < ny_; ++k) result += Basis(j, k, h) * d[i * (ny_ + 1) + k];
        return result;
    }
    std::vector<double> DToFMatrix(const std::vector<double>& d) {
        std::vector<double> result((nx_ + 1) * (ny_ + 1));
        for (size_t i = 1; i < nx_; ++i) {
            for (size_t j = 1; j < ny_; ++j) result[i * (ny_ + 1) + j] = DToF(d, i, j);
        }
        return result;
    }
    std::vector<double> Solve(const std::vector<double>& values) {
        auto c = DToCMatrix(FToDMatrix(values));
        return DToFMatrix(CToDMatrix(c));
    }
private:
    Func eigen_value_;
    int nx_;
    int ny_;
};

template<class Func>
struct Params {
    double G;
    double rho;
    double c;
    double nu;
    Func q;
    double k;
};

double Dx(const std::vector<double>& data, int i, int j, int nx, int ny) {
    return 0.5 * nx * (data[(i + 1) * (ny + 1) + j] - data[(i - 1) * (ny + 1) + j]);
}

double Dy(const std::vector<double>& data, int i, int j, int nx, int ny) {
    return 0.5 * ny * (data[i * (ny + 1) + j + 1] - data[i * (ny + 1) + j - 1]);
}

double Dxx(const std::vector<double>& data, int i, int j, int nx, int ny) {
    return nx * nx * (data[(i + 1) * (ny + 1) + j] - 2 * data[i * (ny + 1) + j]
        + data[(i - 1) * (ny + 1) + j]);
}

double Dyy(const std::vector<double>& data, int i, int j, int nx, int ny) {
    return ny * ny * (data[i * (ny + 1) + j + 1] - 2 * data[i * (ny + 1) + j]
        + data[i * (ny + 1) + j - 1]);
}

template<class Func>
std::vector<double> RightPartForT(std::vector<double>& wP, std::vector<double>& psiP,
    std::vector<double>& TP, int nx, int ny, int M, Params<Func> params) {
    std::vector<double> b((nx + 1) * (ny + 1));
    for (int i = 1; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            b[i * (ny + 1) + j] = TP[(ny + 1) * i + j] * M -  Dy(psiP, i, j, nx, ny)
                * Dx(TP, i, j, nx, ny) + Dx(psiP, i, j, nx, ny) * Dy(TP, i, j, nx, ny)
                + params.q(i, j) / params.rho / params.c;
        }
    }
    return b;
}

template<class Func>
std::vector<double> RightPartForW(std::vector<double>& T, std::vector<double>& wP,
    std::vector<double>& psiP, int nx, int ny, int M, Params<Func> params) {
    std::vector<double> b((nx + 1) * (ny + 1));
    for (int i = 1; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            b[i * (ny + 1) + j] = wP[(ny + 1) * i + j] * M -  Dy(psiP, i, j, nx, ny)
                * Dx(wP, i, j, nx, ny) + Dx(psiP, i, j, nx, ny) * Dy(wP, i, j, nx, ny)
                + params.G * Dx(T, i, j, nx, ny);
        }
    }
    return b;
}

template<class Func>
std::vector<double> RightPartForPsi(std::vector<double>& w, int nx, int ny,
    int M, Params<Func> params) {
    std::vector<double> b((nx + 1) * (ny + 1));
    for (int i = 1; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            b[i * (ny + 1) + j] = - w[i * (ny + 1) + j];
        }
    }
    return b;
}

template<class Func>
std::vector<double> NewT(std::vector<double>& wP, std::vector<double>& psiP,
    std::vector<double>& TP, int nx, int ny, int M, Params<Func> params) {
    Fourier<Func> algo(nx, ny, [&](int s, int t) -> double {
        return M + params.k * 4 * (nx * nx * sin(M_PI * s / 2 / nx) * sin(M_PI * s / 2 / nx)
            + ny * ny * sin(M_PI * t / 2 / ny) * sin(M_PI * t / 2 / ny));
    });
    return algo.Solve(RightPartForT(wP, psiP, TP, nx, ny, M, params));
}

template<class Func>
std::vector<double> NewW(std::vector<double>& T, std::vector<double>& wP,
    std::vector<double>& psiP, int nx, int ny, int M, Params<Func> params) {
    Fourier<Func> algo(nx, ny, [&](int s, int t) -> double {
        return M + params.nu * 4 * (nx * nx * sin(M_PI * s / 2 / nx) * sin(M_PI * s / 2 / nx)
            + ny * ny * sin(M_PI * t / 2 / ny) * sin(M_PI * t / 2 / ny));
    });
    return algo.Solve(RightPartForW(T, wP, psiP, nx, ny, M, params));
}

template<class Func>
std::vector<double> NewPsi(std::vector<double>& w, int nx, int ny,
    int M, Params<Func> params) {
    Fourier<Func> algo(nx, ny, [&](int s, int t) -> double {
        return 4 * (nx * nx * sin(M_PI * s / 2 / nx) * sin(M_PI * s / 2 / nx)
            + ny * ny * sin(M_PI * t / 2 / ny) * sin(M_PI * t / 2 / ny));
    });
    return algo.Solve(RightPartForPsi(w, nx, ny, M, params));
}

void PrintVector(std::vector<double>& data, int nx, int ny) {
    for (int i = 0; i < ny + 1; ++i) {
        for (int j = 0; j < ny + 1; ++j) std::cout << data[i * (ny + 1) + j] << " ";
        std::cout << std::endl;
    }
}

template<class Func>
double CheckFirst(std::vector<std::vector<double>>& ws, std::vector<std::vector<double>>& psis,
    std::vector<std::vector<double>>& Ts, int nx, int ny, int M, Params<Func> params) {
    double residual = 0;
    std::ofstream out("residual1.txt");
    for (int moment = 0; moment < M; ++moment) {
        double current_residual = 0;
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                double difference = fabs((ws[moment + 1][i * (ny + 1) + j]
                    - ws[moment][i * (ny + 1) + j]) * M - params.nu * Dxx(ws[moment + 1], i, j, nx, ny)
                    - params.nu * Dyy(ws[moment + 1], i, j, nx, ny) + Dy(psis[moment], i, j, nx, ny)
                    * Dx(ws[moment], i, j, nx, ny) - Dx(psis[moment], i, j, nx, ny)
                    * Dy(ws[moment], i, j, nx, ny) - params.G * Dx(Ts[moment + 1], i, j, nx, ny));
                residual = std::max(difference, residual);
                current_residual = std::max(difference, current_residual);
            }
        }
        out << moment << " " << -log10(current_residual) << std::endl;
    }
    return residual;
}

template<class Func>
double CheckSecond(std::vector<std::vector<double>>& ws, std::vector<std::vector<double>>& psis,
    std::vector<std::vector<double>>& Ts, int nx, int ny, int M, Params<Func> params) {
    double residual = 0;
    std::ofstream out("residual2.txt");
    for (int moment = 0; moment < M; ++moment) {
        double current_residual = 0;
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                double difference = fabs((Ts[moment + 1][i * (ny + 1) + j]
                    - Ts[moment][i * (ny + 1) + j]) * M - params.k * Dxx(Ts[moment + 1], i, j, nx, ny)
                    - params.k * Dyy(Ts[moment + 1], i, j, nx, ny) + Dy(psis[moment], i, j, nx, ny)
                    * Dx(Ts[moment], i, j, nx, ny) - Dx(psis[moment], i, j, nx, ny)
                    * Dy(Ts[moment], i, j, nx, ny) - params.q(i, j) / params.rho / params.c);
                residual = std::max(difference, residual);
                current_residual = std::max(difference, current_residual);
            }
        }
        out << moment << " " << -log10(current_residual) << std::endl;
    }
    return residual;
}

template<class Func>
double CheckThird(std::vector<std::vector<double>>& ws, std::vector<std::vector<double>>& psis,
    std::vector<std::vector<double>>& Ts, int nx, int ny, int M, Params<Func> params) {
    double residual = 0;
    std::ofstream out("residual3.txt");
    for (int moment = 0; moment < M; ++moment) {
        double current_residual = 0;
        for (int i = 1; i < nx; ++i) {
            for (int j = 1; j < ny; ++j) {
                double difference = fabs(ws[moment][i * (ny + 1) + j]
                    - Dxx(psis[moment], i, j, nx, ny) - Dyy(psis[moment], i, j, nx, ny));
                residual = std::max(difference, residual);
                current_residual = std::max(difference, current_residual);
            }
        }
        out << moment << " " << -log10(current_residual) << std::endl;
    }
    return residual;
}

void Log(std::vector<std::vector<double>>& psis, std::vector<std::vector<double>>& Ts,
    int nx, int ny, int M) {
    std::ofstream out("graph.txt");
    for (int moment = 0; moment < M; ++moment) {
        for (double i = 1; i < nx; ++i) {
            for (double j = 1; j < ny; ++j) {
                out << i / nx << " " << j / ny << " " << Dy(psis[moment], i, j, nx, ny) << " "
                    << - Dx(psis[moment], i, j, nx, ny) << " " << Ts[moment][i * (ny + 1) + j] << std::endl;
            }
        }
        out << std::endl;
        out << std::endl;
    }
}

template<class Func>
void Solve(int nx, int ny, int M, Params<Func> params) {
    std::vector<std::vector<double>> ws(M + 1, std::vector<double>((nx + 1) * (ny + 1)));
    std::vector<std::vector<double>> psis(M + 1, std::vector<double>((nx + 1) * (ny + 1)));
    std::vector<std::vector<double>> Ts(M + 1, std::vector<double>((nx + 1) * (ny + 1)));
    for (int moment = 0; moment < M; ++moment) {
        Ts[moment + 1] = NewT(ws[moment], psis[moment], Ts[moment], nx, ny, M, params);
        ws[moment + 1] = NewW(Ts[moment + 1], ws[moment], psis[moment], nx, ny, M, params);
        psis[moment + 1] = NewPsi(ws[moment + 1], nx, ny, M, params);
    }
    std::cout << CheckFirst(ws, psis, Ts, nx, ny, M, params) << std::endl;
    std::cout << CheckSecond(ws, psis, Ts, nx, ny, M, params) << std::endl;
    std::cout << CheckThird(ws, psis, Ts, nx, ny, M, params) << std::endl;
    Log(psis, Ts, nx, ny, std::min(M, 80));
}


double Q(int i, int j) {
    return 100;
}

int main() {
    int nx, ny, M;
    std::cin >> nx >> ny >> M;
    Params<std::function<double(int, int)>> params;
    params.G = 30;
    params.c = 1;
    params.rho = 1;
    params.nu = 3;
    params.k = 3;
    params.q = Q;
    Solve(nx, ny, M, params);
    return 0;
}
