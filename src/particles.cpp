#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <array>
#include <cstdlib>
#include <cstdio>

#include "partition.h"
#include "constants.h"
#include "particles.h"


static double triangle_area(std::array<std::array<double, 3>, 3> v);

template <typename order_iterator, typename value_iterator>
static void apply_order(order_iterator order_begin, order_iterator order_end, value_iterator v_begin);
    
Particles::Particles(const class Molecule& mol, const struct Params& params)
    : molecule_(mol), params_(params)
{
    Particles::generate_particles(params_.mesh_, params_.mesh_density_, params_.mesh_probe_radius_);
    Particles::compute_source_term(params_.phys_eps_solute_);
    
    source_charge_.assign(num_, 0.);
    source_charge_dx_.assign(num_, 0.);
    source_charge_dy_.assign(num_, 0.);
    source_charge_dz_.assign(num_, 0.);
    
    target_charge_.assign(num_, 0.);
    target_charge_dx_.assign(num_, 0.);
    target_charge_dy_.assign(num_, 0.);
    target_charge_dz_.assign(num_, 0.);
    
    order_.resize(num_);
    std::iota(order_.begin(), order_.end(), 0);
    
    potential_.resize(2 * num_);
};


void Particles::generate_particles(Params::Mesh mesh, double mesh_density, double probe_radius)
{
    std::ofstream NS_param_file("surfaceConfiguration.prm");
    
    NS_param_file << "Grid_scale = "                                 << mesh_density     << std::endl;
    NS_param_file << "Grid_perfil = "                                << 90.0             << std::endl;
    NS_param_file << "XYZR_FileName = "                              << "molecule.xyzr"  << std::endl;
    NS_param_file << "Build_epsilon_maps = "                         << "false"          << std::endl;
    NS_param_file << "Build_status_map = "                           << "false"          << std::endl;
    NS_param_file << "Save_Mesh_MSMS_Format = "                      << "true"           << std::endl;
    NS_param_file << "Compute_Vertex_Normals = "                     << "true"           << std::endl;

    if (mesh == Params::Mesh::SES)  NS_param_file                    << "Surface = ses"  << std::endl;
    if (mesh == Params::Mesh::SKIN) NS_param_file                    << "Surface = skin" << std::endl;
    
    NS_param_file << "Smooth_Mesh = "                                << "true"           << std::endl;
    NS_param_file << "Skin_Surface_Parameter = "                     << 0.45             << std::endl;
    NS_param_file << "Cavity_Detection_Filling = "                   << "false"          << std::endl;
    NS_param_file << "Conditional_Volume_Filling_Value = "           << 11.4             << std::endl;
    NS_param_file << "Keep_Water_Shaped_Cavities = "                 << "false"          << std::endl;
    NS_param_file << "Probe_Radius = "                               << probe_radius     << std::endl;
    NS_param_file << "Accurate_Triangulation = "                     << "true"           << std::endl;
    NS_param_file << "Triangulation = "                              << "true"           << std::endl;
    NS_param_file << "Check_duplicated_vertices = "                  << "true"           << std::endl;
    NS_param_file << "Save_Status_map = "                            << "false"          << std::endl;
    NS_param_file << "Save_PovRay = "                                << "false"          << std::endl;
    NS_param_file << "Max_ses_patches_per_auxiliary_grid_2d_cell = " << 800              << std::endl;

    NS_param_file.close();
    
#ifdef _WIN32
    std::system("NanoShaper.exe");
#else
    std::system("NanoShaper");
#endif

    std::remove("stderror.txt");
    std::remove("surfaceConfiguration.prm");
    std::remove("triangleAreas.txt");
    std::remove("exposed.xyz");
    std::remove("exposedIndices.txt");
    
    std::string line;
    
    // Read in the vert file
    std::ifstream vert_file("triangulatedSurf.vert");
    // Throw away first two lines
    std::getline(vert_file, line);
    std::getline(vert_file, line);
    
    std::getline(vert_file, line);
    std::size_t num_vertices = std::stoul(line);
    
    std::vector<double> vertex_x;
    std::vector<double> vertex_y;
    std::vector<double> vertex_z;
    std::vector<double> vertex_normal_x;
    std::vector<double> vertex_normal_y;
    std::vector<double> vertex_normal_z;
    
    vertex_x.reserve(num_vertices);
    vertex_y.reserve(num_vertices);
    vertex_z.reserve(num_vertices);
    vertex_normal_x.reserve(num_vertices);
    vertex_normal_y.reserve(num_vertices);
    vertex_normal_z.reserve(num_vertices);
    
    while (std::getline(vert_file, line)) {
        
        std::istringstream iss(line);
        std::vector<std::string> tokenized_line{std::istream_iterator<std::string> {iss},
                                                std::istream_iterator<std::string> {} };
                                                
        vertex_x.push_back(std::stod(tokenized_line[0]));
        vertex_y.push_back(std::stod(tokenized_line[1]));
        vertex_z.push_back(std::stod(tokenized_line[2]));
        
        vertex_normal_x.push_back(std::stod(tokenized_line[3]));
        vertex_normal_y.push_back(std::stod(tokenized_line[4]));
        vertex_normal_z.push_back(std::stod(tokenized_line[5]));
    }
    
    vert_file.close();
    
    // Read in the face file
    std::ifstream face_file("triangulatedSurf.face");
    // Throw away first two lines
    std::getline(face_file, line);
    std::getline(face_file, line);
    
    std::getline(face_file, line);
    std::size_t num_faces = std::stoul(line);
    std::vector<std::size_t> face_x;
    std::vector<std::size_t> face_y;
    std::vector<std::size_t> face_z;
    
    face_x.reserve(num_faces);
    face_y.reserve(num_faces);
    face_z.reserve(num_faces);
    
    while (std::getline(face_file, line)) {
        
        std::istringstream iss(line);
        std::vector<std::string> tokenized_line{std::istream_iterator<std::string> {iss},
                                                std::istream_iterator<std::string> {} };
                                                
        face_x.push_back(std::stoul(tokenized_line[0]));
        face_y.push_back(std::stoul(tokenized_line[1]));
        face_z.push_back(std::stoul(tokenized_line[2]));
    }
    
    face_file.close();
    
    std::remove("molecule.xyzr");
    std::remove("triangulatedSurf.vert");
    std::remove("triangulatedSurf.face");
    
    /*
    // Removing bad triangles
    std::size_t num_removed = 0;
    
    for (std::size_t i = 0; i < num_faces; ++i) {
        
        std::size_t iface[3] {face_x[i], face_y[i], face_z[i]};
        double xx[3] {0., 0., 0.};
        double r[3][3];

        for (int ii = 0; ii < 3; ++ii) {
        
            r[0][ii] = vertex_x[iface[ii]-1];
            r[1][ii] = vertex_y[iface[ii]-1];
            r[2][ii] = vertex_z[iface[ii]-1];
            
            xx[0] += 1.0/3.0 * r[0][ii];
            xx[1] += 1.0/3.0 * r[1][ii];
            xx[2] += 1.0/3.0 * r[2][ii];
        }

        double triangle_area = triangleArea(r);
        
        if (triangle_area < 1e-5) {
            face_x.erase(face_x.begin() + i - num_removed);
            face_y.erase(face_y.begin() + i - num_removed);
            face_z.erase(face_z.begin() + i - num_removed);
            num_removed++;
            continue;
        }

        for (std::size_t j = i-10; (j >= 0 && j < i); ++j) {
        
            std::size_t jface[3] {face_x[j], face_y[j], face_z[j]};
            double yy[3] {0., 0., 0.};

            for (int jj = 0; jj < 3; ++jj) {
            
                r[0][jj] = vertex_x[jface[jj]-1];
                r[1][jj] = vertex_y[jface[jj]-1];
                r[2][jj] = vertex_z[jface[jj]-1];
                
                yy[0] += 1.0/3.0 * r[0][jj];
                yy[1] += 1.0/3.0 * r[1][jj];
                yy[2] += 1.0/3.0 * r[2][jj];
            }

            double triangle_distance = 0.0;
            for (int jj = 0; jj < 3; ++jj) {
                triangle_distance += (xx[jj]-yy[jj]) * (xx[jj]-yy[jj]);
            }
            triangle_distance = std::sqrt(triangle_distance);

            if (triangle_distance < 1e-5) {
                face_x.erase(face_x.begin() + i - num_removed);
                face_y.erase(face_y.begin() + i - num_removed);
                face_z.erase(face_z.begin() + i - num_removed);
                num_removed++;
                break;
            }
        }
    }
    */
    
    num_ = num_vertices;
    x_ = std::move(vertex_x);
    y_ = std::move(vertex_y);
    z_ = std::move(vertex_z);
    
    nx_ = std::move(vertex_normal_x);
    ny_ = std::move(vertex_normal_y);
    nz_ = std::move(vertex_normal_z);
    
    area_.assign(num_, 0.);
    
    for (std::size_t i = 0; i < num_faces; ++i) {
    
        std::array<std::size_t, 3> iface {face_x[i], face_y[i], face_z[i]};
        std::array<std::array<double, 3>, 3> r; 
        
        for (int ii = 0; ii < 3; ++ii) {
            r[0][ii] = x_[iface[ii]-1];
            r[1][ii] = y_[iface[ii]-1];
            r[2][ii] = z_[iface[ii]-1];
        }

        for (int j = 0; j < 3; ++j) {
            area_[iface[j]-1] += triangle_area(r);
        }
    }
    
    std::transform(area_.begin(), area_.end(), area_.begin(),
                   [](double x) { return x / 3.; } );
    surface_area_ = std::accumulate(area_.begin(), area_.end(), decltype(area_)::value_type(0));
    std::cout << "Surface area is " << surface_area_ << std::endl;
}



void Particles::compute_source_term(double eps_solute)
{
/* this computes the source term where
 * S1=sum(qk*G0)/e1 S2=sim(qk*G0')/e1 */

    source_term_.assign(2 * num_, 0.);
    
    const double* __restrict__ particles_x_ptr = x_.data();
    const double* __restrict__ particles_y_ptr = y_.data();
    const double* __restrict__ particles_z_ptr = z_.data();
    
    const double* __restrict__ particles_nx_ptr = nx_.data();
    const double* __restrict__ particles_ny_ptr = ny_.data();
    const double* __restrict__ particles_nz_ptr = nz_.data();
    
    const double* __restrict__ molecule_coords_ptr = molecule_.coords_ptr();
    const double* __restrict__ molecule_charge_ptr = molecule_.charge_ptr();
    
    double* __restrict__ particles_source_term_ptr = source_term_.data();

    for (std::size_t i = 0; i < num_; ++i) {
        for (std::size_t j = 0; j < molecule_.num_atoms(); ++j) {

  /* r_s = distance of charge position to triangular */
            double x_dist = molecule_coords_ptr[3*j + 0] - particles_x_ptr[i];
            double y_dist = molecule_coords_ptr[3*j + 1] - particles_y_ptr[i];
            double z_dist = molecule_coords_ptr[3*j + 2] - particles_z_ptr[i];
            double dist   = std::sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);

  /* cos_theta = <tr_q,r_s>/||r_s||_2 */
            double cos_theta = (particles_nx_ptr[i] * x_dist
                              + particles_ny_ptr[i] * y_dist
                              + particles_nz_ptr[i] * z_dist) / dist;

  /* G0 = 1/(4pi*||r_s||_2) */
            double G0 = constants::ONE_OVER_4PI / dist;

  /* G1 = cos_theta*G0/||r_s||_2 */
            double G1 = cos_theta * G0 / dist;

  /* update source term */
            particles_source_term_ptr[i]        += molecule_charge_ptr[j] * G0 / eps_solute;
            particles_source_term_ptr[num_ + i] += molecule_charge_ptr[j] * G1 / eps_solute;
        }
    }
}


double Particles::compute_solvation_energy(std::vector<double>& potential) const
{
    double eps = params_.phys_eps_;
    double kappa = params_.phys_kappa_;
    double solvation_energy = 0.;
    
    const double* __restrict__ particles_x_ptr = x_.data();
    const double* __restrict__ particles_y_ptr = y_.data();
    const double* __restrict__ particles_z_ptr = z_.data();
    
    const double* __restrict__ particles_nx_ptr = nx_.data();
    const double* __restrict__ particles_ny_ptr = ny_.data();
    const double* __restrict__ particles_nz_ptr = nz_.data();
    
    const double* __restrict__ particles_area_ptr = area_.data();
    
    const double* __restrict__ molecule_coords_ptr = molecule_.coords_ptr();
    const double* __restrict__ molecule_charge_ptr = molecule_.charge_ptr();
    
    const double* __restrict__ potential_ptr = potential.data();
    
    for (std::size_t i = 0; i < num_; ++i) {
        for (std::size_t j = 0; j < molecule_.num_atoms(); ++j) {
        
            double x_dist = particles_x_ptr[i] - molecule_coords_ptr[3*j + 0];
            double y_dist = particles_y_ptr[i] - molecule_coords_ptr[3*j + 1];
            double z_dist = particles_z_ptr[i] - molecule_coords_ptr[3*j + 2];
            double dist   = std::sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);

            double cos_theta   = (particles_nx_ptr[i] * x_dist
                                + particles_ny_ptr[i] * y_dist
                                + particles_nz_ptr[i] * z_dist) / dist;

            double kappa_r     = kappa * dist;
            double exp_kappa_r = std::exp(-kappa_r);

            double G0 = constants::ONE_OVER_4PI / dist;
            double Gk = exp_kappa_r * G0;
            double G1 = cos_theta * G0 / dist;
            double G2 = G1 * (1.0 + kappa_r) * exp_kappa_r;
        
            double L1 = G1 - eps * G2;
            double L2 = G0 - Gk;

            solvation_energy += molecule_charge_ptr[j] * particles_area_ptr[i]
                              * (L1 * potential_ptr[i] + L2 * potential_ptr[num_ + i]);
        }
    }

    return solvation_energy;
}


const std::array<double, 6> Particles::bounds(std::size_t begin, std::size_t end) const
{
    auto x_min_max = std::minmax_element(x_.begin() + begin, x_.begin() + end);
    auto y_min_max = std::minmax_element(y_.begin() + begin, y_.begin() + end);
    auto z_min_max = std::minmax_element(z_.begin() + begin, z_.begin() + end);
    
    return std::array<double, 6> {*x_min_max.first, *x_min_max.second,
                                  *y_min_max.first, *y_min_max.second,
                                  *z_min_max.first, *z_min_max.second};
}


int Particles::partition_8(std::size_t begin, std::size_t end, std::array<std::size_t, 16>& partitioned_bounds)
{
    int num_children = 1;
    
    partitioned_bounds[0] = begin;
    partitioned_bounds[1] = end;
    
    auto bounds = Particles::bounds(begin, end);
    
    double x_len = bounds[1] - bounds[0];
    double y_len = bounds[3] - bounds[2];
    double z_len = bounds[5] - bounds[4];
    
    double x_mid = (bounds[1] + bounds[0]) / 2.;
    double y_mid = (bounds[3] + bounds[2]) / 2.;
    double z_mid = (bounds[5] + bounds[4]) / 2.;
    
    double max_len = x_len;
    if (max_len < y_len) max_len = y_len;
    if (max_len < z_len) max_len = z_len;

    double critical_len = max_len / std::sqrt(2.);
    
    bool divide_x = false;
    bool divide_y = false;
    bool divide_z = false;
    
    if (x_len > critical_len) divide_x = true;
    if (y_len > critical_len) divide_y = true;
    if (z_len > critical_len) divide_z = true;

    if (divide_x) {

//  This, unfortunately, does not quite work, but it should be something like this to reorder them in an STL way
//
//        std::vector<size_t> reorder_vec(ind[0][1] - ind[0][0] + 1);
//        std::iota(reorder_vec.begin(), reorder_vec.end(), ind[0][0]);
//
//        auto pivot = std::partition(reorder_vec.begin(), reorder_vec.end(),
//                                    [&x, &x_mid, &ind](size_t elem){ return x[elem] < x_mid; });
//
//        std::transform(reorder_vec.begin(),reorder_vec.end(),reorder_vec.begin(),
//                       [&ind](size_t i){ return i-ind[0][0]; });
//
//        reorder_inplace_destructive(reorder_vec.begin(), reorder_vec.end(),
//                        orderarr.begin()+ind[0][0], x.begin()+ind[0][0], y.begin()+ind[0][0], z.begin()+ind[0][0]);
//
//        ind[1][0] = *pivot + ind[0][0];
//        ind[1][1] = ind[0][1];
//        ind[0][1] = *pivot + ind[0][0] - 1;

        std::size_t node_begin = partitioned_bounds[0];
        std::size_t node_end   = partitioned_bounds[1];

        std::size_t pivot_idx = partition<double>(x_.data(), y_.data(), z_.data(), order_.data(),
                                                  node_begin, node_end, x_mid);
        
        partitioned_bounds[2] = pivot_idx;
        partitioned_bounds[3] = partitioned_bounds[1];
        partitioned_bounds[1] = pivot_idx;

        num_children *= 2;
    }

    if (divide_y) {

        for (int i = 0; i < num_children; ++i) {
        
            std::size_t node_begin = partitioned_bounds[2*i + 0];
            std::size_t node_end   = partitioned_bounds[2*i + 1];
        
            std::size_t pivot_idx = partition<double>(y_.data(), x_.data(), z_.data(), order_.data(),
                                                      node_begin, node_end, y_mid);

            partitioned_bounds[2 * (num_children + i) + 0] = pivot_idx;
            partitioned_bounds[2 * (num_children + i) + 1] = partitioned_bounds[2*i + 1];
            partitioned_bounds[2*i + 1] = pivot_idx;
        }

        num_children *= 2;
    }

    if (divide_z) {

        for (int i = 0; i < num_children; ++i) {
        
            std::size_t node_begin = partitioned_bounds[2*i + 0];
            std::size_t node_end   = partitioned_bounds[2*i + 1];
        
            std::size_t pivot_idx = partition<double>(z_.data(), x_.data(), y_.data(), order_.data(),
                                                      node_begin, node_end, z_mid);

            partitioned_bounds[2 * (num_children + i) + 0] = pivot_idx;
            partitioned_bounds[2 * (num_children + i) + 1] = partitioned_bounds[2*i + 1];
            partitioned_bounds[2*i + 1] = pivot_idx;
        }

        num_children *= 2;
    }

    return num_children;

}


void Particles::reorder()
{
    apply_order(order_.begin(), order_.end(), nx_.begin());
    apply_order(order_.begin(), order_.end(), ny_.begin());
    apply_order(order_.begin(), order_.end(), nz_.begin());
    
    apply_order(order_.begin(), order_.end(), area_.begin());
    apply_order(order_.begin(), order_.end(), source_term_.begin());
    apply_order(order_.begin(), order_.end(), source_term_.begin() + num_);
}


void Particles::unorder(std::vector<double>& potential)
{
    apply_order(order_.begin(), order_.end(), x_.begin());
    apply_order(order_.begin(), order_.end(), y_.begin());
    apply_order(order_.begin(), order_.end(), z_.begin());
    
    apply_order(order_.begin(), order_.end(), nx_.begin());
    apply_order(order_.begin(), order_.end(), ny_.begin());
    apply_order(order_.begin(), order_.end(), nz_.begin());
    
    apply_order(order_.begin(), order_.end(), area_.begin());
    apply_order(order_.begin(), order_.end(), source_term_.begin());
    apply_order(order_.begin(), order_.end(), source_term_.begin() + num_);
    
    apply_order(order_.begin(), order_.end(), potential.begin());
    apply_order(order_.begin(), order_.end(), potential.begin() + num_);
}


void Particles::compute_charges(const std::vector<double>& potential)
{
    for (std::size_t i = 0; i < num_; ++i) {
        target_charge_   [i] = constants::ONE_OVER_4PI;
        target_charge_dx_[i] = constants::ONE_OVER_4PI * nx_[i];
        target_charge_dy_[i] = constants::ONE_OVER_4PI * ny_[i];
        target_charge_dz_[i] = constants::ONE_OVER_4PI * nz_[i];
        
        source_charge_   [i] =          area_[i] * potential[num_ + i];
        source_charge_dx_[i] = nx_[i] * area_[i] * potential[i];
        source_charge_dy_[i] = ny_[i] * area_[i] * potential[i];
        source_charge_dz_[i] = nz_[i] * area_[i] * potential[i];
    }
}


void Particles::compute_charges(const double* potential)
{
    for (std::size_t i = 0; i < num_; ++i) {
        target_charge_   [i] = constants::ONE_OVER_4PI;
        target_charge_dx_[i] = constants::ONE_OVER_4PI * nx_[i];
        target_charge_dy_[i] = constants::ONE_OVER_4PI * ny_[i];
        target_charge_dz_[i] = constants::ONE_OVER_4PI * nz_[i];
        
        source_charge_   [i] =          area_[i] * potential[num_ + i];
        source_charge_dx_[i] = nx_[i] * area_[i] * potential[i];
        source_charge_dy_[i] = ny_[i] * area_[i] * potential[i];
        source_charge_dz_[i] = nz_[i] * area_[i] * potential[i];
    }
}


static double triangle_area(std::array<std::array<double, 3>, 3> v)
{
    std::array<double, 3> a, b, c;

    for (int i = 0; i < 3; ++i) {
        a[i] = v[i][0] - v[i][1];
        b[i] = v[i][0] - v[i][2];
        c[i] = v[i][1] - v[i][2];
    }

    double aa = std::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    double bb = std::sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);
    double cc = std::sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);

    double ss = 0.5 * (aa + bb + cc);
    return std::sqrt(ss * (ss-aa) * (ss-bb) * (ss-cc));
}


template <typename order_iterator, typename value_iterator>
static void apply_order(order_iterator order_begin, order_iterator order_end, value_iterator v_begin)
{
    using value_t = typename std::iterator_traits< value_iterator >::value_type;
    using index_t = typename std::iterator_traits< order_iterator >::value_type;
    
    auto v_end = v_begin + std::distance(order_begin, order_end) + 1;
    std::vector<value_t> tmp(v_begin, v_end);

    std::for_each(order_begin, order_end,
                  [&tmp, &v_begin](index_t idx){ *v_begin = tmp[idx]; std::advance(v_begin, 1); });
}
