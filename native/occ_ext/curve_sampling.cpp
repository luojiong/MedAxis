#include "curve_sampling.h"

#include <Geom_BSplineCurve.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <GCPnts_UniformAbscissa.hxx>
#include <GeomLProp_CLProps.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>

#include <stdexcept>

namespace medaxis
{
namespace occ_ext
{

namespace
{

/// Fit a B-spline curve through the supplied points.
Handle(Geom_BSplineCurve) fit_curve(const std::vector<Point3>& control_points)
{
    if (control_points.size() < 2)
    {
        throw std::runtime_error("fit_curve: need at least 2 points");
    }

    TColgp_Array1OfPnt pts(1, static_cast<Standard_Integer>(control_points.size()));
    for (std::size_t i = 0; i < control_points.size(); ++i)
    {
        pts.SetValue(static_cast<Standard_Integer>(i + 1),
                     gp_Pnt(control_points[i][0], control_points[i][1], control_points[i][2]));
    }

    GeomAPI_PointsToBSpline approximator(pts, 3 /* degree */);
    if (!approximator.IsDone())
    {
        throw std::runtime_error("fit_curve: B-spline fitting failed");
    }
    return approximator.Curve();
}

Point3 to_array(const gp_Pnt& p)
{
    return {p.X(), p.Y(), p.Z()};
}

Point3 to_array(const gp_Dir& d)
{
    return {d.X(), d.Y(), d.Z()};
}

} // namespace

std::vector<Point3> sample_curve_equal_arc_length(const std::vector<Point3>& control_points,
                                                  int num_samples)
{
    if (num_samples < 2)
    {
        throw std::runtime_error("sample_curve_equal_arc_length: need >= 2 samples");
    }

    Handle(Geom_BSplineCurve) curve = fit_curve(control_points);

    GeomAdaptor_Curve adaptor(curve);
    GCPnts_UniformAbscissa sampler(adaptor, num_samples, curve->FirstParameter(),
                                   curve->LastParameter(), 1.0e-8);
    if (!sampler.IsDone())
    {
        throw std::runtime_error("sample_curve_equal_arc_length: uniform abscissa failed");
    }

    std::vector<Point3> samples;
    samples.reserve(num_samples);
    for (Standard_Integer i = 1; i <= sampler.NbPoints(); ++i)
    {
        const Standard_Real u = sampler.Parameter(i);
        samples.push_back(to_array(curve->Value(u)));
    }
    return samples;
}

double curve_arc_length(const std::vector<Point3>& control_points)
{
    Handle(Geom_BSplineCurve) curve = fit_curve(control_points);
    GeomAdaptor_Curve adaptor(curve);
    // OCCT 7.8: GCPnts_AbscissaType is an enum; Length() lives on AbscissaPoint.
    return GCPnts_AbscissaPoint::Length(adaptor);
}

std::vector<FrenetFrame> frenet_frames(const std::vector<Point3>& control_points,
                                       const std::vector<double>& parameters)
{
    Handle(Geom_BSplineCurve) curve = fit_curve(control_points);

    const Standard_Real u0 = curve->FirstParameter();
    const Standard_Real u1 = curve->LastParameter();

    std::vector<FrenetFrame> frames;
    frames.reserve(parameters.size());

    for (double t : parameters)
    {
        if (t < 0.0 || t > 1.0)
        {
            throw std::runtime_error("frenet_frames: parameter out of [0, 1]");
        }
        const Standard_Real u = u0 + t * (u1 - u0);

        // Order 2 for curvature-dependent frame quantities.
        GeomLProp_CLProps props(curve, u, 2, 1.0e-8);

        FrenetFrame frame;
        frame.point = to_array(props.Value());

        gp_Dir d;
        if (props.IsTangentDefined())
        {
            props.Tangent(d);
            frame.tangent = to_array(d);
        }
        else
        {
            frame.tangent = {0.0, 0.0, 1.0};
        }

        // OCCT 7.8 CLProps exposes Curvature()/Normal() but no
        // IsNormalDefined(); a normal exists iff curvature is non-zero.
        if (props.Curvature() > 1.0e-12)
        {
            props.Normal(d);
            frame.normal = to_array(d);
        }
        else
        {
            frame.normal = {0.0, 1.0, 0.0};
        }

        // OCCT 7.8 CLProps has no Binormal(); compute T x N.
        frame.binormal = {frame.tangent[1] * frame.normal[2] - frame.tangent[2] * frame.normal[1],
                          frame.tangent[2] * frame.normal[0] - frame.tangent[0] * frame.normal[2],
                          frame.tangent[0] * frame.normal[1] - frame.tangent[1] * frame.normal[0]};

        frames.push_back(frame);
    }

    return frames;
}

} // namespace occ_ext
} // namespace medaxis
