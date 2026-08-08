/*=========================================================================

  vtkMedAxisCPRFilter.cxx

  Skeleton implementation of Curved Planar Reformation.

  Algorithm:
    1. Walk the curve point by point (one output column per curve point).
    2. At each point build an orthonormal frame:
         T = tangent (finite differences along the curve)
         N = user-supplied normal (or heuristic fallback)
         B = T x N
    3. Reslice a 1 x NumRows strip of the input volume in the plane
       spanned by (N, T) centered on the curve point using vtkImageReslice.
    4. Stitch all strips into the output image (X = arc index, Y = offset).

=========================================================================*/
#include "vtkMedAxisCPRFilter.h"

#include "vtkFloatArray.h"
#include "vtkImageData.h"
#include "vtkImageReslice.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <cmath>

vtkStandardNewMacro(vtkMedAxisCPRFilter);

//----------------------------------------------------------------------------
vtkMedAxisCPRFilter::vtkMedAxisCPRFilter() = default;

//----------------------------------------------------------------------------
vtkMedAxisCPRFilter::~vtkMedAxisCPRFilter()
{
    if (this->CurvePoints)
    {
        this->CurvePoints->Delete();
    }
    if (this->CurveNormals)
    {
        this->CurveNormals->Delete();
    }
}

//----------------------------------------------------------------------------
void vtkMedAxisCPRFilter::SetCurvePoints(vtkPoints* points)
{
    if (this->CurvePoints == points)
    {
        return;
    }
    if (this->CurvePoints)
    {
        this->CurvePoints->Delete();
    }
    this->CurvePoints = points;
    if (this->CurvePoints)
    {
        this->CurvePoints->Register(this);
    }
    this->Modified();
}

//----------------------------------------------------------------------------
void vtkMedAxisCPRFilter::SetCurveNormals(vtkFloatArray* normals)
{
    if (this->CurveNormals == normals)
    {
        return;
    }
    if (this->CurveNormals)
    {
        this->CurveNormals->Delete();
    }
    this->CurveNormals = normals;
    if (this->CurveNormals)
    {
        this->CurveNormals->Register(this);
    }
    this->Modified();
}

//----------------------------------------------------------------------------
int vtkMedAxisCPRFilter::RequestInformation(
    vtkInformation* vtkNotUsed(request),
    vtkInformationVector** vtkNotUsed(inputVector),
    vtkInformationVector* outputVector)
{
    if (!this->CurvePoints || this->CurvePoints->GetNumberOfPoints() < 2)
    {
        vtkErrorMacro("CPR requires at least 2 curve points.");
        return 0;
    }

    const vtkIdType numCols = this->CurvePoints->GetNumberOfPoints();
    int extent[6] = {0, static_cast<int>(numCols) - 1, 0, this->NumRows - 1, 0, 0};

    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent, 6);

    const double rowSpacing = (2.0 * this->Thickness) / (this->NumRows - 1);
    double spacing[3] = {1.0, rowSpacing, 1.0};
    outInfo->Set(vtkDataObject::SPACING(), spacing, 3);

    double origin[3] = {0.0, -this->Thickness, 0.0};
    outInfo->Set(vtkDataObject::ORIGIN(), origin, 3);

    return 1;
}

//----------------------------------------------------------------------------
namespace
{
void Normalize(double v[3])
{
    const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1.0e-12)
    {
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
    }
}

void Cross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
} // namespace

//----------------------------------------------------------------------------
int vtkMedAxisCPRFilter::RequestData(
    vtkInformation* vtkNotUsed(request),
    vtkInformationVector** inputVector,
    vtkInformationVector* outputVector)
{
    vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
    vtkInformation* outInfo = outputVector->GetInformationObject(0);

    vtkImageData* input =
        vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkImageData* output =
        vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

    if (!input || !output || !this->CurvePoints)
    {
        vtkErrorMacro("CPR: missing input, output or curve points.");
        return 0;
    }

    const vtkIdType numPoints = this->CurvePoints->GetNumberOfPoints();
    if (numPoints < 2)
    {
        vtkErrorMacro("CPR: need at least 2 curve points.");
        return 0;
    }

    const double rowSpacing = (2.0 * this->Thickness) / (this->NumRows - 1);

    // Pre-allocate output scalars (copy scalar type from input).
    output->SetDimensions(static_cast<int>(numPoints), this->NumRows, 1);
    output->AllocateScalars(input->GetScalarType(), input->GetNumberOfScalarComponents());

    // ------------------------------------------------------------------
    // For each curve point: build the local frame, reslice one column.
    // ------------------------------------------------------------------
    for (vtkIdType i = 0; i < numPoints; ++i)
    {
        // Tangent: central finite differences.
        double pPrev[3], pNext[3], pCur[3];
        this->CurvePoints->GetPoint(i > 0 ? i - 1 : 0, pPrev);
        this->CurvePoints->GetPoint(i < numPoints - 1 ? i + 1 : numPoints - 1, pNext);
        this->CurvePoints->GetPoint(i, pCur);

        double tangent[3] = {pNext[0] - pPrev[0], pNext[1] - pPrev[1], pNext[2] - pPrev[2]};
        Normalize(tangent);

        // Normal: user supplied or heuristic (least-aligned world axis).
        double normal[3];
        if (this->CurveNormals && this->CurveNormals->GetNumberOfTuples() > i)
        {
            float n[3];
            this->CurveNormals->GetTypedTuple(i, n);
            normal[0] = n[0];
            normal[1] = n[1];
            normal[2] = n[2];
        }
        else
        {
            // Fallback: pick the world axis least parallel to the tangent.
            const double ax = std::fabs(tangent[0]);
            const double ay = std::fabs(tangent[1]);
            const double az = std::fabs(tangent[2]);
            if (ax <= ay && ax <= az)      normal[0] = 1, normal[1] = 0, normal[2] = 0;
            else if (ay <= az)             normal[0] = 0, normal[1] = 1, normal[2] = 0;
            else                           normal[0] = 0, normal[1] = 0, normal[2] = 1;
        }
        Normalize(normal);

        // Orthonormalize normal against tangent, then derive binormal.
        const double dot = normal[0] * tangent[0] + normal[1] * tangent[1] +
                           normal[2] * tangent[2];
        normal[0] -= dot * tangent[0];
        normal[1] -= dot * tangent[1];
        normal[2] -= dot * tangent[2];
        Normalize(normal);

        double binormal[3];
        Cross(tangent, normal, binormal);
        Normalize(binormal);
        (void)binormal; // reserved for future slab-CPR modes

        // Reslice axes: X -> curve direction, Y -> normal, origin at point.
        vtkNew<vtkMatrix4x4> axes;
        for (int k = 0; k < 3; ++k)
        {
            axes->SetElement(k, 0, tangent[k]);
            axes->SetElement(k, 1, normal[k]);
            axes->SetElement(k, 2, binormal[k]);
            axes->SetElement(k, 3, pCur[k]);
        }

        vtkNew<vtkImageReslice> reslice;
        reslice->SetInputData(input);
        reslice->SetResliceAxes(axes);
        reslice->SetInterpolationModeToLinear();
        reslice->SetOutputOrigin(0.0, -this->Thickness, 0.0);
        reslice->SetOutputSpacing(1.0, rowSpacing, 1.0);
        reslice->SetOutputExtent(0, 0, 0, this->NumRows - 1, 0, 0);
        reslice->Update();

        // Stitch: copy the 1 x NumRows strip into output column i.
        vtkImageData* strip = reslice->GetOutput();
        for (int row = 0; row < this->NumRows; ++row)
        {
            output->GetPointData()->GetScalars()->SetTuple(
                i + static_cast<vtkIdType>(row) * numPoints,
                strip->GetPointData()->GetScalars()->GetTuple(row));
        }

        if (i % 50 == 0)
        {
            this->UpdateProgress(static_cast<double>(i) / numPoints);
        }
    }

    return 1;
}

//----------------------------------------------------------------------------
void vtkMedAxisCPRFilter::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << indent << "Thickness: " << this->Thickness << "\n";
    os << indent << "NumRows: " << this->NumRows << "\n";
    os << indent << "CurvePoints: "
       << (this->CurvePoints ? this->CurvePoints->GetNumberOfPoints() : 0) << "\n";
}
