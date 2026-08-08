/*=========================================================================

  vtkMedAxisResliceFilter.cxx

  Skeleton implementation: delegates the heavy lifting to vtkImageReslice.
  A future phase may replace the internals with a custom threaded kernel
  (e.g. GPU-backed reslicing) without changing the public API.

=========================================================================*/
#include "vtkMedAxisResliceFilter.h"

#include "vtkImageData.h"
#include "vtkImageReslice.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"

vtkStandardNewMacro(vtkMedAxisResliceFilter);

//----------------------------------------------------------------------------
vtkMedAxisResliceFilter::vtkMedAxisResliceFilter()
{
    this->ResliceAxes = nullptr;
}

//----------------------------------------------------------------------------
vtkMedAxisResliceFilter::~vtkMedAxisResliceFilter()
{
    if (this->ResliceAxes)
    {
        this->ResliceAxes->Delete();
        this->ResliceAxes = nullptr;
    }
}

//----------------------------------------------------------------------------
void vtkMedAxisResliceFilter::SetResliceAxes(vtkMatrix4x4* axes)
{
    if (this->ResliceAxes == axes)
    {
        return;
    }
    if (this->ResliceAxes)
    {
        this->ResliceAxes->Delete();
    }
    this->ResliceAxes = axes;
    if (this->ResliceAxes)
    {
        this->ResliceAxes->Register(this);
    }
    this->Modified();
}

//----------------------------------------------------------------------------
int vtkMedAxisResliceFilter::RequestInformation(
    vtkInformation* vtkNotUsed(request),
    vtkInformationVector** vtkNotUsed(inputVector),
    vtkInformationVector* outputVector)
{
    // Skeleton: let the downstream decide the extent for now. A full
    // implementation computes the output extent from the input bounds
    // projected through the reslice axes.
    vtkInformation* outInfo = outputVector->GetInformationObject(0);
    (void)outInfo;
    return 1;
}

//----------------------------------------------------------------------------
int vtkMedAxisResliceFilter::RequestData(
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

    if (!input || !output)
    {
        vtkErrorMacro("Missing input or output image data.");
        return 0;
    }

    // Delegate to vtkImageReslice for the actual interpolation.
    vtkNew<vtkImageReslice> reslice;
    reslice->SetInputData(input);

    if (this->ResliceAxes)
    {
        reslice->SetResliceAxes(this->ResliceAxes);
    }
    else
    {
        // Identity axes: reslice in the native orientation.
        vtkNew<vtkMatrix4x4> identity;
        identity->Identity();
        reslice->SetResliceAxes(identity);
    }

    switch (this->InterpolationMode)
    {
        case NEAREST_NEIGHBOR:
            reslice->SetInterpolationModeToNearestNeighbor();
            break;
        case TRILINEAR:
            reslice->SetInterpolationModeToLinear();
            break;
        case BICUBIC:
            reslice->SetInterpolationModeToCubic();
            break;
        case SINC:
            // VTK 9.6 has no sinc reslice mode; cubic is the closest.
            reslice->SetInterpolationModeToCubic();
            break;
        default:
            reslice->SetInterpolationModeToLinear();
            break;
    }

    // Slab thickness: average over N slices worth of input spacing.
    if (this->SlabThickness > 1)
    {
        reslice->SetSlabModeToMean();
        reslice->SetSlabNumberOfSlices(static_cast<int>(this->SlabThickness));
        reslice->SetSlabSliceSpacingFraction(1.0);
    }

    reslice->SetOutputSpacing(this->OutputSpacing, this->OutputSpacing, this->OutputSpacing);
    reslice->SetOptimization(1);
    reslice->Update();

    output->ShallowCopy(reslice->GetOutput());
    return 1;
}

//----------------------------------------------------------------------------
void vtkMedAxisResliceFilter::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os, indent);
    os << indent << "InterpolationMode: " << this->InterpolationMode << "\n";
    os << indent << "SlabThickness: " << this->SlabThickness << "\n";
    os << indent << "OutputSpacing: " << this->OutputSpacing << "\n";
    os << indent << "ResliceAxes: " << (this->ResliceAxes ? "(set)" : "(none)") << "\n";
}
