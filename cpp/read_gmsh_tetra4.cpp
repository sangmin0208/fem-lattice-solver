#define _POSIX_C_SOURCE 200809L
#include <ctime>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <tuple>

namespace py = pybind11;

std::tuple<py::array_t<double>, py::array_t<int>> read_gmsh_tetra4_mesh(const std::string& filename, double scale = 1.0) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<std::array<double, 4>> nodes;
    std::vector<std::array<int, 4>> elements;
    std::string line;

    while (std::getline(file, line)) {
        // Handle $Nodes section
        if (line.find("$Nodes") != std::string::npos) {
            std::getline(file, line);
            int num_nodes = std::stoi(line);
            nodes.reserve(num_nodes);
            for (int i = 0; i < num_nodes; ++i) {
                int id;
                double x, y, z;
                file >> id >> x >> y >> z;
                nodes.push_back({(double)id, x * scale, y * scale, z * scale});
            }
        }

        // Handle $Elements section
        if (line.find("$Elements") != std::string::npos) {
            std::getline(file, line);
            int num_elements = std::stoi(line);
            for (int i = 0; i < num_elements; ++i) {
                std::string element_line;
                std::getline(file >> std::ws, element_line);
                std::stringstream ss(element_line);

                int id, type, num_tags;
                ss >> id >> type >> num_tags;

                if (type == 4) {  // 4-node tetrahedron
                    for (int j = 0; j < num_tags; ++j) { int dummy; ss >> dummy; }

                    std::array<int, 4> tet;
                    for (int j = 0; j < 4; ++j) {
                        ss >> tet[j];
                        tet[j] -= 1;  // zero-based indexing
                    }
                    elements.push_back(tet);
                }
            }
            break;  // done
        }
    }

    // Convert to numpy arrays
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

PYBIND11_MODULE(read_gmsh_cpp, m) {
    m.def("read_gmsh_tetra4_mesh", &read_gmsh_tetra4_mesh, "Fast Gmsh Tetra4 Reader",
          py::arg("filename"), py::arg("scale") = 1.0);
}
