#define _POSIX_C_SOURCE 200809L
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <sys/stat.h>

namespace py = pybind11;

// Load ASCII Gmsh v2 mesh and extract Tetra4 nodes & elements
std::tuple<py::array_t<double>, py::array_t<int>>
read_gmsh_tetra4(const std::string& filename, double scale = 1.0) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open file: " + filename);

    struct stat st;
    stat(filename.c_str(), &st);
    size_t fsize = st.st_size;
    std::vector<char> buffer(fsize + 1);
    size_t bytes_read = fread(buffer.data(), 1, fsize, fp);
    if (bytes_read != fsize) throw std::runtime_error("Incomplete fread");
    buffer[fsize] = '\0';
    std::fclose(fp);

    char* p = buffer.data();
    char* end = buffer.data() + fsize;

    std::vector<std::array<double, 4>> nodes;
    std::vector<std::array<int, 4>> elements;

    while (p < end) {
        if (strncmp(p, "$Nodes", 6) == 0) {
            while (*p != '\n') ++p; ++p;
            int num_nodes = std::strtol(p, &p, 10);
            nodes.reserve(num_nodes);
            for (int i = 0; i < num_nodes; ++i) {
                int id = std::strtol(p, &p, 10);
                double x = std::strtod(p, &p);
                double y = std::strtod(p, &p);
                double z = std::strtod(p, &p);
                nodes.push_back({(double)id, x * scale, y * scale, z * scale});
            }
        } else if (strncmp(p, "$Elements", 9) == 0) {
            while (*p != '\n') ++p; ++p;
            int num_elements = std::strtol(p, &p, 10);
            elements.reserve(num_elements / 2);
            for (int i = 0; i < num_elements; ++i) {
                int id = std::strtol(p, &p, 10);
                int type = std::strtol(p, &p, 10);
                int num_tags = std::strtol(p, &p, 10);
                for (int j = 0; j < num_tags; ++j) std::strtol(p, &p, 10);
                if (type == 4) {
                    std::array<int, 4> tet;
                    for (int j = 0; j < 4; ++j) tet[j] = std::strtol(p, &p, 10) - 1;
                    elements.push_back(tet);
                } else {
                    while (*p != '\n' && p < end) ++p; ++p;
                }
            }
            break;
        } else {
            while (*p != '\n' && p < end) ++p; ++p;
        }
    }

    py::array_t<double> node_arr({(int)nodes.size(), 4});
    py::array_t<int> elem_arr({(int)elements.size(), 4});
    auto n_ptr = node_arr.mutable_unchecked<2>();
    auto e_ptr = elem_arr.mutable_unchecked<2>();

    for (size_t i = 0; i < nodes.size(); ++i)
        for (int j = 0; j < 4; ++j)
            n_ptr(i, j) = nodes[i][j];

    for (size_t i = 0; i < elements.size(); ++i)
        for (int j = 0; j < 4; ++j)
            e_ptr(i, j) = elements[i][j];

    return {node_arr, elem_arr};
}

// Assemble global stiffness triplets (row, col, val) from Tetra4 mesh
py::tuple assemble_tetra4_stiffness_triplets(py::array_t<double> nodes,
                                              py::array_t<int> elements,
                                              double lambda,
                                              double mu) {
    auto n = nodes.unchecked<2>();
    auto e = elements.unchecked<2>();
    int n_elem = e.shape(0);

    py::array_t<int> row({n_elem * 144});
    py::array_t<int> col({n_elem * 144});
    py::array_t<double> data({n_elem * 144});

    auto r = row.mutable_data();
    auto c = col.mutable_data();
    auto d = data.mutable_data();

    const double dNdxi[4][3] = {
        {-1.0, -1.0, -1.0}, {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},    {0.0, 0.0, 1.0}
    };
    
    #pragma omp parallel for
    for (int ei = 0; ei < n_elem; ++ei) {
        int ni[4]; double Xe[4][3];
        for (int j = 0; j < 4; ++j) {
            ni[j] = e(ei, j);
            Xe[j][0] = n(ni[j], 1);
            Xe[j][1] = n(ni[j], 2);
            Xe[j][2] = n(ni[j], 3);
        }

        double J[3][3] = {};
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    J[i][j] += Xe[a][i] * dNdxi[a][j];

        double detJ = J[0][0]*(J[1][1]*J[2][2] - J[1][2]*J[2][1])
                    - J[0][1]*(J[1][0]*J[2][2] - J[1][2]*J[2][0])
                    + J[0][2]*(J[1][0]*J[2][1] - J[1][1]*J[2][0]);

        if (std::abs(detJ) < 1e-12) continue;
        double volume = detJ / 6.0;

        double invJ[3][3], invdet = 1.0 / detJ;
        invJ[0][0] =  (J[1][1]*J[2][2] - J[1][2]*J[2][1]) * invdet;
        invJ[0][1] = -(J[0][1]*J[2][2] - J[0][2]*J[2][1]) * invdet;
        invJ[0][2] =  (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * invdet;
        invJ[1][0] = -(J[1][0]*J[2][2] - J[1][2]*J[2][0]) * invdet;
        invJ[1][1] =  (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * invdet;
        invJ[1][2] = -(J[0][0]*J[1][2] - J[0][2]*J[1][0]) * invdet;
        invJ[2][0] =  (J[1][0]*J[2][1] - J[1][1]*J[2][0]) * invdet;
        invJ[2][1] = -(J[0][0]*J[2][1] - J[0][1]*J[2][0]) * invdet;
        invJ[2][2] =  (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * invdet;

        double dNdX[4][3] = {};
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    dNdX[a][i] += dNdxi[a][j] * invJ[j][i];

        double B[6][12] = {};
        for (int a = 0; a < 4; ++a) {
            int ia = a * 3;
            B[0][ia+0] = dNdX[a][0];
            B[1][ia+1] = dNdX[a][1];
            B[2][ia+2] = dNdX[a][2];
            B[3][ia+0] = dNdX[a][1]; B[3][ia+1] = dNdX[a][0];
            B[4][ia+0] = dNdX[a][2]; B[4][ia+2] = dNdX[a][0];
            B[5][ia+1] = dNdX[a][2]; B[5][ia+2] = dNdX[a][1];
        }

        double C[6][6] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                C[i][j] = (i == j) ? lambda + 2 * mu : lambda;
        for (int i = 3; i < 6; ++i) C[i][i] = mu;

        double CB[6][12] = {};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 6; ++k)
                    CB[i][j] += C[i][k] * B[k][j];

        double Ke[12][12] = {};
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j)
                for (int k = 0; k < 6; ++k)
                    Ke[i][j] += B[k][i] * CB[k][j] * volume;

        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j) {
                int idx = ei * 144 + i * 12 + j;
                r[idx] = ni[i / 3] * 3 + (i % 3);
                c[idx] = ni[j / 3] * 3 + (j % 3);
                d[idx] = Ke[i][j];
            }
    }

    return py::make_tuple(row, col, data);
}

PYBIND11_MODULE(tetra4_fem_core, m) {
    m.def("read_gmsh_tetra4_mesh", &read_gmsh_tetra4, "Read Tetra4 mesh from ASCII .msh", py::arg("filename"), py::arg("scale") = 1.0);
    m.def("assemble_tetra4_triplets", &assemble_tetra4_stiffness_triplets, "Assemble triplets from Tetra4 mesh", py::arg("nodes"), py::arg("elements"), py::arg("lambda"), py::arg("mu"));
}
