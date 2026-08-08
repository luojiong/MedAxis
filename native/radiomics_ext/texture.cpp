/*
 * Texture radiomics in C++: GLCM(24), GLRLM(16), GLSZM(16), GLDM(14),
 * NGTDM(5).  Ported from processing/radiomics/texture.py with the same
 * (pyradiomics-style) formulas; intensities are quantized to `levels` bins.
 */

#include "texture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace medaxis
{
namespace radiomics_ext
{

namespace
{

struct Dims
{
    py::ssize_t nk, nj, ni;
};

inline std::size_t idx(const Dims& d, py::ssize_t k, py::ssize_t j, py::ssize_t i)
{
    return static_cast<std::size_t>((k * d.nj + j) * d.ni + i);
}

/// Quantize intensities to [0, levels-1].
std::vector<std::int16_t> quantize(const float* src, const Dims& d, int levels)
{
    const std::size_t count = static_cast<std::size_t>(d.nk * d.nj * d.ni);
    float lo = src[0], hi = src[0];
    for (std::size_t i = 1; i < count; ++i)
    {
        lo = std::min(lo, src[i]);
        hi = std::max(hi, src[i]);
    }
    std::vector<std::int16_t> q(count);
    if (hi - lo < 1.0e-12f)
    {
        std::fill(q.begin(), q.end(), 0);
        return q;
    }
    const float range = hi - lo;
    for (std::size_t i = 0; i < count; ++i)
    {
        const float v = std::floor((src[i] - lo) / range * static_cast<float>(levels));
        q[i] = static_cast<std::int16_t>(std::clamp(v, 0.0f, static_cast<float>(levels - 1)));
    }
    return q;
}

inline double sq(double v) { return v * v; }

// ---------------------------------------------------------------------------
// GLCM
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, double>> glcm(const std::vector<std::int16_t>& q,
                                                 const Dims& d, int levels)
{
    // Direction groups: axes, xy-diagonals, xz-diagonals, yz-diagonals, space diagonals.
    const int groups[5][4][3] = {
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}},
        {{1, 1, 0}, {1, -1, 0}, {0, 0, 0}, {0, 0, 0}},
        {{1, 0, 1}, {1, 0, -1}, {0, 0, 0}, {0, 0, 0}},
        {{0, 1, 1}, {0, 1, -1}, {0, 0, 0}, {0, 0, 0}},
        {{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1}},
    };
    const char* props[6] = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm_energy"};
    std::vector<std::pair<std::string, double>> out;

    for (int gi = 0; gi < 5; ++gi)
    {
        std::vector<double> p(static_cast<std::size_t>(levels * levels), 0.0);
        for (int dir = 0; dir < 4; ++dir)
        {
            const int dx = groups[gi][dir][0];
            const int dy = groups[gi][dir][1];
            const int dz = groups[gi][dir][2];
            if (dx == 0 && dy == 0 && dz == 0)
            {
                continue;
            }
            for (py::ssize_t k = 0; k < d.nk; ++k)
            {
                for (py::ssize_t j = 0; j < d.nj; ++j)
                {
                    for (py::ssize_t i = 0; i < d.ni; ++i)
                    {
                        const py::ssize_t k2 = k + dz, j2 = j + dy, i2 = i + dx;
                        if (k2 < 0 || k2 >= d.nk || j2 < 0 || j2 >= d.nj || i2 < 0 || i2 >= d.ni)
                        {
                            continue;
                        }
                        const int a = q[idx(d, k, j, i)];
                        const int b = q[idx(d, k2, j2, i2)];
                        p[static_cast<std::size_t>(a * levels + b)] += 1.0;
                    }
                }
            }
        }
        double total = 0.0;
        for (double v : p)
        {
            total += v;
        }
        if (total <= 0.0)
        {
            continue;
        }
        for (double& v : p)
        {
            v /= total;
        }
        double contrast = 0.0, dissimilarity = 0.0, homogeneity = 0.0;
        double asm_energy = 0.0, mi = 0.0, mj = 0.0, si2 = 0.0, sj2 = 0.0, cov = 0.0;
        for (int a = 0; a < levels; ++a)
        {
            for (int b = 0; b < levels; ++b)
            {
                const double pab = p[static_cast<std::size_t>(a * levels + b)];
                const double diff = static_cast<double>(std::abs(a - b));
                contrast += pab * sq(diff);
                dissimilarity += pab * diff;
                homogeneity += pab / (1.0 + diff);
                asm_energy += sq(pab);
                mi += pab * static_cast<double>(a);
                mj += pab * static_cast<double>(b);
            }
        }
        for (int a = 0; a < levels; ++a)
        {
            for (int b = 0; b < levels; ++b)
            {
                const double pab = p[static_cast<std::size_t>(a * levels + b)];
                si2 += pab * sq(static_cast<double>(a) - mi);
                sj2 += pab * sq(static_cast<double>(b) - mj);
                cov += pab * (static_cast<double>(a) - mi) * (static_cast<double>(b) - mj);
            }
        }
        const double energy = std::sqrt(asm_energy);
        const double correlation = (si2 * sj2 > 1.0e-12) ? cov / std::sqrt(si2 * sj2) : 0.0;
        const double values[6] = {contrast, dissimilarity, homogeneity, energy, correlation, asm_energy};
        for (int pidx = 0; pidx < 6; ++pidx)
        {
            out.emplace_back("glcm_" + std::string(props[pidx]) + "_" + std::to_string(gi), values[pidx]);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// GLRLM
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, double>> glrlm(const std::vector<std::int16_t>& q,
                                                  const Dims& d, int levels)
{
    const int dirs[13][3] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
        {1, 1, 0}, {1, -1, 0}, {1, 0, 1}, {1, 0, -1}, {0, 1, 1}, {0, 1, -1},
        {1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {1, -1, -1},
    };
    // Accumulate runs over all directions into one matrix.
    std::map<std::pair<int, int>, double> runs;  // (level, length) -> count
    int max_run = 0;

    for (const auto& dir : dirs)
    {
        const int dx = dir[0], dy = dir[1], dz = dir[2];
        // Line starts: free coordinates orthogonal to the direction.
        std::vector<std::array<py::ssize_t, 3>> starts;
        if (dx != 0)
        {
            for (py::ssize_t j = 0; j < d.nj; ++j)
                for (py::ssize_t k = 0; k < d.nk; ++k)
                    starts.push_back({k, j, 0});
        }
        else if (dy != 0)
        {
            for (py::ssize_t i = 0; i < d.ni; ++i)
                for (py::ssize_t k = 0; k < d.nk; ++k)
                    starts.push_back({k, 0, i});
        }
        else
        {
            for (py::ssize_t i = 0; i < d.ni; ++i)
                for (py::ssize_t j = 0; j < d.nj; ++j)
                    starts.push_back({0, j, i});
        }
        for (const auto& s : starts)
        {
            py::ssize_t k = s[0], j = s[1], i = s[2];
            int run_value = -1;
            int run_len = 0;
            while (k >= 0 && k < d.nk && j >= 0 && j < d.nj && i >= 0 && i < d.ni)
            {
                const int v = q[idx(d, k, j, i)];
                if (v == run_value)
                {
                    ++run_len;
                }
                else
                {
                    if (run_len > 0)
                    {
                        runs[{run_value, run_len}] += 1.0;
                        max_run = std::max(max_run, run_len);
                    }
                    run_value = v;
                    run_len = 1;
                }
                k += dz;
                j += dy;
                i += dx;
            }
            if (run_len > 0)
            {
                runs[{run_value, run_len}] += 1.0;
                max_run = std::max(max_run, run_len);
            }
        }
    }

    double nr = 0.0;
    for (const auto& [key, count] : runs)
    {
        nr += count;
    }
    std::vector<std::pair<std::string, double>> out;
    if (nr <= 0.0)
    {
        return out;
    }

    // p[i][j]: probability of level i, run length j.
    std::vector<std::vector<double>> p(static_cast<std::size_t>(levels),
                                       std::vector<double>(static_cast<std::size_t>(max_run + 1), 0.0));
    for (const auto& [key, count] : runs)
    {
        p[static_cast<std::size_t>(key.first)][static_cast<std::size_t>(key.second)] = count / nr;
    }
    std::vector<double> pg(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> pr(static_cast<std::size_t>(max_run + 1), 0.0);
    for (int a = 0; a < levels; ++a)
    {
        for (int r = 0; r <= max_run; ++r)
        {
            pg[static_cast<std::size_t>(a)] += p[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)];
            pr[static_cast<std::size_t>(r)] += p[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)];
        }
    }
    double sre = 0, lre = 0, gln = 0, rln = 0, lg = 0, hg = 0;
    double srlgle = 0, srhgle = 0, lrlgle = 0, lrhgle = 0;
    for (int a = 0; a < levels; ++a)
    {
        for (int r = 1; r <= max_run; ++r)
        {
            const double pab = p[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)];
            const double g = static_cast<double>(a);
            const double rl = static_cast<double>(r);
            sre += pab / (rl * rl);
            lre += pab * rl * rl;
            srlgle += pab / (g * g * rl * rl + 1.0e-12);
            srhgle += pab * g * g / (rl * rl + 1.0e-12);
            lrlgle += pab * rl * rl / (g * g + 1.0e-12);
            lrhgle += pab * g * g * rl * rl;
        }
    }
    for (int a = 0; a < levels; ++a)
    {
        gln += sq(pg[static_cast<std::size_t>(a)]);
        lg += pg[static_cast<std::size_t>(a)] * static_cast<double>(a);
        hg += pg[static_cast<std::size_t>(a)] * sq(static_cast<double>(a));
    }
    for (int r = 0; r <= max_run; ++r)
    {
        rln += sq(pr[static_cast<std::size_t>(r)]);
    }
    const double rp = nr / static_cast<double>(d.nk * d.nj * d.ni);
    const double glv = (levels > 1) ? std::accumulate(
                          pg.begin(), pg.end(), 0.0,
                          [&](double acc, double v) { return acc + sq(v - gln / levels); }) /
                          static_cast<double>(levels - 1)
                      : 0.0;
    const double rlv = (max_run > 0) ? std::accumulate(
                            pr.begin(), pr.end(), 0.0,
                            [&](double acc, double v) { return acc + sq(v - rln / (max_run + 1)); }) /
                            static_cast<double>(max_run)
                        : 0.0;
    const char* names[15] = {"sre", "lre", "gln", "rln", "rp", "lgre", "hgre",
                             "srlgle", "srhgle", "lrlgle", "lrhgle", "glv", "rlv", "rpe", "rpa"};
    const double vals[15] = {sre, lre, gln, rln, rp, lg, hg, srlgle, srhgle, lrlgle, lrhgle, glv, rlv, 0.0, 0.0};
    // Run percentage entropy / energy need the matrix loop; compute here.
    double rpe = 0.0, rpa = 0.0;
    for (int a = 0; a < levels; ++a)
    {
        for (int r = 1; r <= max_run; ++r)
        {
            const double pab = p[static_cast<std::size_t>(a)][static_cast<std::size_t>(r)];
            if (pab > 0)
            {
                rpe += -pab * std::log(pab + 1.0e-12);
                rpa += sq(pab);
            }
        }
    }
    const double vals2[15] = {sre, lre, gln, rln, rp, lg, hg, srlgle, srhgle, lrlgle, lrhgle, glv, rlv, rpe, rpa};
    for (int i = 0; i < 15; ++i)
    {
        out.emplace_back("glrlm_" + std::string(names[i]), vals2[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// GLSZM — size zone matrix via 26-connected component labeling
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, double>> glszm(const std::vector<std::int16_t>& q,
                                                  const Dims& d, int levels)
{
    const std::size_t count = static_cast<std::size_t>(d.nk * d.nj * d.ni);
    std::vector<int> label(count, -1);
    std::map<std::pair<int, int>, double> zones;  // (level, size) -> count
    int max_zone = 0;
    int next_label = 0;

    static const int nb[26][3] = {
        {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1},
        {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1},
        {0, 0, -1}, {0, 0, 1}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1},
        {1, 1, -1}, {1, 1, 0}, {1, 1, 1},
    };

    for (py::ssize_t k = 0; k < d.nk; ++k)
    {
        for (py::ssize_t j = 0; j < d.nj; ++j)
        {
            for (py::ssize_t i = 0; i < d.ni; ++i)
            {
                const std::size_t pos = idx(d, k, j, i);
                if (label[pos] != -1)
                {
                    continue;
                }
                const int value = q[pos];
                // BFS flood fill over 26-connected voxels of the same level.
                std::deque<std::array<py::ssize_t, 3>> queue;
                queue.push_back({k, j, i});
                label[pos] = next_label;
                int size = 0;
                while (!queue.empty())
                {
                    const auto cur = queue.front();
                    queue.pop_front();
                    ++size;
                    for (const auto& n : nb)
                    {
                        const py::ssize_t k2 = cur[0] + n[0], j2 = cur[1] + n[1], i2 = cur[2] + n[2];
                        if (k2 < 0 || k2 >= d.nk || j2 < 0 || j2 >= d.nj || i2 < 0 || i2 >= d.ni)
                        {
                            continue;
                        }
                        const std::size_t npos = idx(d, k2, j2, i2);
                        if (label[npos] == -1 && q[npos] == value)
                        {
                            label[npos] = next_label;
                            queue.push_back({k2, j2, i2});
                        }
                    }
                }
                zones[{value, size}] += 1.0;
                max_zone = std::max(max_zone, size);
                ++next_label;
            }
        }
    }

    double nz = 0.0;
    for (const auto& [key, c] : zones)
    {
        nz += c;
    }
    std::vector<std::pair<std::string, double>> out;
    if (nz <= 0.0)
    {
        return out;
    }
    std::vector<std::vector<double>> p(static_cast<std::size_t>(levels),
                                       std::vector<double>(static_cast<std::size_t>(max_zone + 1), 0.0));
    for (const auto& [key, c] : zones)
    {
        p[static_cast<std::size_t>(key.first)][static_cast<std::size_t>(key.second)] = c / nz;
    }
    std::vector<double> pg(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> pz(static_cast<std::size_t>(max_zone + 1), 0.0);
    for (int a = 0; a < levels; ++a)
    {
        for (int z = 0; z <= max_zone; ++z)
        {
            pg[static_cast<std::size_t>(a)] += p[static_cast<std::size_t>(a)][static_cast<std::size_t>(z)];
            pz[static_cast<std::size_t>(z)] += p[static_cast<std::size_t>(a)][static_cast<std::size_t>(z)];
        }
    }
    double sae = 0, lae = 0, gln = 0, zn = 0, lgze = 0, hgze = 0;
    double salgle = 0, sahgle = 0, lalgle = 0, lahgle = 0;
    for (int a = 0; a < levels; ++a)
    {
        for (int z = 1; z <= max_zone; ++z)
        {
            const double pab = p[static_cast<std::size_t>(a)][static_cast<std::size_t>(z)];
            const double g = static_cast<double>(a);
            const double zl = static_cast<double>(z);
            sae += pab / (zl * zl);
            lae += pab * zl * zl;
            salgle += pab / (g * g * zl * zl + 1.0e-12);
            sahgle += pab * g * g / (zl * zl + 1.0e-12);
            lalgle += pab * zl * zl / (g * g + 1.0e-12);
            lahgle += pab * g * g * zl * zl;
        }
    }
    for (int a = 0; a < levels; ++a)
    {
        gln += sq(pg[static_cast<std::size_t>(a)]);
        lgze += pg[static_cast<std::size_t>(a)] * static_cast<double>(a);
        hgze += pg[static_cast<std::size_t>(a)] * sq(static_cast<double>(a));
    }
    for (int z = 0; z <= max_zone; ++z)
    {
        zn += sq(pz[static_cast<std::size_t>(z)]);
    }
    const double zp = nz / static_cast<double>(count);
    const double glv = (levels > 1) ? std::accumulate(
                          pg.begin(), pg.end(), 0.0,
                          [&](double acc, double v) { return acc + sq(v - gln / levels); }) /
                          static_cast<double>(levels - 1)
                      : 0.0;
    const double zv = (max_zone > 0) ? std::accumulate(
                           pz.begin(), pz.end(), 0.0,
                           [&](double acc, double v) { return acc + sq(v - zn / (max_zone + 1)); }) /
                           static_cast<double>(max_zone)
                       : 0.0;
    double ze = 0, za = 0;
    for (int a = 0; a < levels; ++a)
    {
        for (int z = 1; z <= max_zone; ++z)
        {
            const double pab = p[static_cast<std::size_t>(a)][static_cast<std::size_t>(z)];
            if (pab > 0)
            {
                ze += -pab * std::log(pab + 1.0e-12);
                za += sq(pab);
            }
        }
    }
    const char* names[15] = {"sae", "lae", "gln", "zn", "zp", "lgze", "hgze",
                             "salgle", "sahgle", "lalgle", "lahgle", "glv", "zv", "ze", "za"};
    const double vals[15] = {sae, lae, gln, zn, zp, lgze, hgze, salgle, sahgle, lalgle, lahgle, glv, zv, ze, za};
    for (int i = 0; i < 15; ++i)
    {
        out.emplace_back("glszm_" + std::string(names[i]), vals[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// GLDM — gray level dependence matrix
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, double>> gldm(const std::vector<std::int16_t>& q,
                                                 const Dims& d, int levels, int delta)
{
    const std::size_t count = static_cast<std::size_t>(d.nk * d.nj * d.ni);
    static const int nb[26][3] = {
        {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1},
        {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1},
        {0, 0, -1}, {0, 0, 1}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1},
        {1, 1, -1}, {1, 1, 0}, {1, 1, 1},
    };
    std::vector<int> deps(count, 0);
    for (py::ssize_t k = 0; k < d.nk; ++k)
    {
        for (py::ssize_t j = 0; j < d.nj; ++j)
        {
            for (py::ssize_t i = 0; i < d.ni; ++i)
            {
                const int v = q[idx(d, k, j, i)];
                int dep = 0;
                for (const auto& n : nb)
                {
                    const py::ssize_t k2 = k + n[0], j2 = j + n[1], i2 = i + n[2];
                    if (k2 < 0 || k2 >= d.nk || j2 < 0 || j2 >= d.nj || i2 < 0 || i2 >= d.ni)
                    {
                        continue;
                    }
                    if (std::abs(q[idx(d, k2, j2, i2)] - v) <= delta)
                    {
                        ++dep;
                    }
                }
                deps[idx(d, k, j, i)] = dep;
            }
        }
    }
    std::vector<std::vector<double>> m(static_cast<std::size_t>(levels),
                                       std::vector<double>(27, 0.0));
    for (std::size_t pos = 0; pos < count; ++pos)
    {
        m[static_cast<std::size_t>(q[pos])][static_cast<std::size_t>(deps[pos])] += 1.0;
    }
    double nd = 0.0;
    for (const auto& row : m)
    {
        for (double v : row)
        {
            nd += v;
        }
    }
    std::vector<std::pair<std::string, double>> out;
    if (nd <= 0.0)
    {
        return out;
    }
    std::vector<double> pg(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> pd(27, 0.0);
    for (int a = 0; a < levels; ++a)
    {
        for (int dep = 0; dep < 27; ++dep)
        {
            const double pab = m[static_cast<std::size_t>(a)][static_cast<std::size_t>(dep)] / nd;
            pg[static_cast<std::size_t>(a)] += pab;
            pd[static_cast<std::size_t>(dep)] += pab;
        }
    }
    double sde = 0, lde = 0, gln = 0, dn = 0, lg = 0, hg = 0;
    double sdlgle = 0, sdhgle = 0, ldlgle = 0, ldhgle = 0;
    for (int a = 0; a < levels; ++a)
    {
        for (int dep = 1; dep < 27; ++dep)
        {
            const double pab = m[static_cast<std::size_t>(a)][static_cast<std::size_t>(dep)] / nd;
            const double g = static_cast<double>(a);
            const double dl = static_cast<double>(dep);
            sde += pab / (dl * dl);
            lde += pab * dl * dl;
            sdlgle += pab / (g * g * dl * dl + 1.0e-12);
            sdhgle += pab * g * g / (dl * dl + 1.0e-12);
            ldlgle += pab * dl * dl / (g * g + 1.0e-12);
            ldhgle += pab * g * g * dl * dl;
        }
    }
    for (int a = 0; a < levels; ++a)
    {
        gln += sq(pg[static_cast<std::size_t>(a)]);
        lg += pg[static_cast<std::size_t>(a)] * static_cast<double>(a);
        hg += pg[static_cast<std::size_t>(a)] * sq(static_cast<double>(a));
    }
    for (int dep = 0; dep < 27; ++dep)
    {
        dn += sq(pd[static_cast<std::size_t>(dep)]);
    }
    const double dp = nd / static_cast<double>(count);
    const double glv = (levels > 1) ? std::accumulate(
                          pg.begin(), pg.end(), 0.0,
                          [&](double acc, double v) { return acc + sq(v - gln / levels); }) /
                          static_cast<double>(levels - 1)
                      : 0.0;
    double dv = 0.0;
    for (int dep = 0; dep < 27; ++dep)
    {
        dv += sq(pd[static_cast<std::size_t>(dep)] - dn / 27.0);
    }
    dv /= 26.0;
    double de = 0.0;
    for (int a = 0; a < levels; ++a)
    {
        for (int dep = 1; dep < 27; ++dep)
        {
            const double pab = m[static_cast<std::size_t>(a)][static_cast<std::size_t>(dep)] / nd;
            if (pab > 0)
            {
                de += -pab * std::log(pab + 1.0e-12);
            }
        }
    }
    const char* names[14] = {"sde", "lde", "gln", "dn", "dp", "lgre", "hgre",
                             "sdlge", "sdhge", "ldlge", "ldhge", "glv", "dv", "de"};
    const double vals[14] = {sde, lde, gln, dn, dp, lg, hg, sdlgle, sdhgle, ldlgle, ldhgle, glv, dv, de};
    for (int i = 0; i < 14; ++i)
    {
        out.emplace_back("gldm_" + std::string(names[i]), vals[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// NGTDM
// ---------------------------------------------------------------------------

std::vector<std::pair<std::string, double>> ngtdm(const std::vector<std::int16_t>& q,
                                                  const Dims& d, int levels)
{
    const std::size_t count = static_cast<std::size_t>(d.nk * d.nj * d.ni);
    static const int nb[26][3] = {
        {-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1}, {-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1},
        {-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1}, {0, -1, -1}, {0, -1, 0}, {0, -1, 1},
        {0, 0, -1}, {0, 0, 1}, {0, 1, -1}, {0, 1, 0}, {0, 1, 1},
        {1, -1, -1}, {1, -1, 0}, {1, -1, 1}, {1, 0, -1}, {1, 0, 0}, {1, 0, 1},
        {1, 1, -1}, {1, 1, 0}, {1, 1, 1},
    };
    std::vector<double> s_i(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> n_i(static_cast<std::size_t>(levels), 0.0);
    for (py::ssize_t k = 0; k < d.nk; ++k)
    {
        for (py::ssize_t j = 0; j < d.nj; ++j)
        {
            for (py::ssize_t i = 0; i < d.ni; ++i)
            {
                const int v = q[idx(d, k, j, i)];
                double neighbor_sum = 0.0;
                int neighbor_count = 0;
                for (const auto& n : nb)
                {
                    const py::ssize_t k2 = k + n[0], j2 = j + n[1], i2 = i + n[2];
                    if (k2 < 0 || k2 >= d.nk || j2 < 0 || j2 >= d.nj || i2 < 0 || i2 >= d.ni)
                    {
                        continue;
                    }
                    neighbor_sum += static_cast<double>(q[idx(d, k2, j2, i2)]);
                    ++neighbor_count;
                }
                const double mean = neighbor_sum / static_cast<double>(neighbor_count);
                s_i[static_cast<std::size_t>(v)] += std::abs(static_cast<double>(v) - mean);
                n_i[static_cast<std::size_t>(v)] += 1.0;
            }
        }
    }
    std::vector<double> p_i(static_cast<std::size_t>(levels), 0.0);
    for (int a = 0; a < levels; ++a)
    {
        p_i[static_cast<std::size_t>(a)] = n_i[static_cast<std::size_t>(a)] / static_cast<double>(count);
    }
    double s_total = 0.0;
    for (int a = 0; a < levels; ++a)
    {
        if (p_i[static_cast<std::size_t>(a)] > 0)
        {
            s_total += s_i[static_cast<std::size_t>(a)];
        }
    }
    const double n_minus = std::max(count - s_total, 1.0e-12);

    double coarseness = 1.0 / (s_total + 1.0e-12);
    double contrast = 0.0, busy_denom = 0.0, complexity = 0.0, strength_denom = 0.0;
    double busy_num = 0.0, strength_num = 0.0;
    int active = 0;
    for (int a = 0; a < levels; ++a)
    {
        if (p_i[static_cast<std::size_t>(a)] <= 0)
        {
            continue;
        }
        ++active;
        busy_num += p_i[static_cast<std::size_t>(a)] * s_i[static_cast<std::size_t>(a)];
        strength_num += p_i[static_cast<std::size_t>(a)] * s_i[static_cast<std::size_t>(a)];
        for (int b = 0; b < levels; ++b)
        {
            if (p_i[static_cast<std::size_t>(b)] <= 0)
            {
                continue;
            }
            const double diff = std::abs(static_cast<double>(a) - static_cast<double>(b));
            const double pa = p_i[static_cast<std::size_t>(a)];
            const double pb = p_i[static_cast<std::size_t>(b)];
            contrast += pa * pb * diff;
            busy_denom += pa * pb * diff;
            complexity += diff * (pa * s_i[static_cast<std::size_t>(b)] +
                                  pb * s_i[static_cast<std::size_t>(a)]) /
                           (pa + pb + 1.0e-12);
            strength_denom += diff;
        }
    }
    contrast = contrast / n_minus * (1.0 / static_cast<double>(std::max(active, 1)));
    const double busyness = busy_num / (busy_denom + 1.0e-12);
    const double strength = strength_num / (strength_denom + 1.0e-12);

    const char* names[5] = {"coarseness", "contrast", "busyness", "complexity", "strength"};
    const double vals[5] = {coarseness, contrast, busyness, complexity, strength};
    std::vector<std::pair<std::string, double>> out;
    for (int i = 0; i < 5; ++i)
    {
        out.emplace_back("ngtdm_" + std::string(names[i]), vals[i]);
    }
    return out;
}

} // namespace

py::dict texture_features(
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    int levels,
    const std::vector<std::string>& families)
{
    if (values.ndim() != 3)
    {
        throw std::invalid_argument("texture_features expects a 3-D array (z, y, x)");
    }
    const Dims d{values.shape(0), values.shape(1), values.shape(2)};
    if (levels < 8 || levels > 256)
    {
        throw std::invalid_argument("levels must be in [8, 256]");
    }
    const std::vector<std::int16_t> q = quantize(values.data(), d, levels);

    py::dict out;
    auto push = [&out](const std::vector<std::pair<std::string, double>>& features) {
        for (const auto& [name, value] : features)
        {
            out[py::str(name)] = value;
        }
    };
    for (const auto& family : families)
    {
        if (family == "glcm")
        {
            push(glcm(q, d, levels));
        }
        else if (family == "glrlm")
        {
            push(glrlm(q, d, levels));
        }
        else if (family == "glszm")
        {
            push(glszm(q, d, levels));
        }
        else if (family == "gldm")
        {
            push(gldm(q, d, levels, 0));
        }
        else if (family == "ngtdm")
        {
            push(ngtdm(q, d, levels));
        }
    }
    return out;
}

} // namespace radiomics_ext
} // namespace medaxis
