/*=========================================================================

  medAxisBrushFilter.hxx

  Applies all accumulated strokes to the output label image. Each stroke
  only iterates voxels inside its bounding box, so painting remains fast
  even on very large volumes.

=========================================================================*/
#ifndef medAxisBrushFilter_hxx
#define medAxisBrushFilter_hxx

#include "medAxisBrushFilter.h"

#include "itkImageRegionConstIterator.h"

#include <cmath>

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
BrushFilter<TInputImage, TOutputImage>::BrushFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
BrushFilter<TInputImage, TOutputImage>::GenerateData()
{
    // Start from a copy of the input label image.
    this->AllocateOutputs();
    typename TOutputImage::Pointer output = this->GetOutput();

    const auto region = output->GetLargestPossibleRegion();
    const auto& size = region.GetSize();
    const auto* input = this->GetInput();

    {
        itk::ImageRegionConstIterator<TInputImage> inIt(input, region);
        for (inIt.GoToBegin(); !inIt.IsAtEnd(); ++inIt)
        {
            output->SetPixel(inIt.GetIndex(),
                             static_cast<OutputPixelType>(inIt.Get()));
        }
    }

    using IndexValueType = typename IndexType::IndexValueType;

    for (const Stroke& stroke : m_Strokes)
    {
        const auto r = static_cast<IndexValueType>(std::ceil(stroke.radius));
        const double sigma = std::max(1.0e-6, stroke.radius * 0.5);

        // Bounding box clamped to the image.
        IndexType lo, hi;
        for (unsigned int d = 0; d < ImageDimension; ++d)
        {
            lo[d] = std::max<IndexValueType>(stroke.center[d] - r, 0);
            hi[d] = std::min<IndexValueType>(stroke.center[d] + r,
                                             static_cast<IndexValueType>(size[d]) - 1);
        }

        IndexType idx;
        for (idx[2] = (ImageDimension > 2 ? lo[2] : 0);
             idx[2] <= (ImageDimension > 2 ? hi[2] : 0);
             ++idx[2])
        {
            for (idx[1] = lo[1]; idx[1] <= hi[1]; ++idx[1])
            {
                for (idx[0] = lo[0]; idx[0] <= hi[0]; ++idx[0])
                {
                    // Squared distance from stroke center in voxels.
                    double dist2 = 0.0;
                    for (unsigned int d = 0; d < ImageDimension; ++d)
                    {
                        const double delta = static_cast<double>(idx[d] - stroke.center[d]);
                        dist2 += delta * delta;
                    }

                    bool inside = false;
                    switch (stroke.shape)
                    {
                        case BrushShape::Sphere:
                            inside = dist2 <= stroke.radius * stroke.radius;
                            break;
                        case BrushShape::Box:
                            inside = true; // bounding box is the brush
                            break;
                        case BrushShape::Gaussian3D:
                        {
                            const double weight = std::exp(-dist2 / (2.0 * sigma * sigma));
                            inside = weight >= m_GaussianCutoff;
                            break;
                        }
                    }

                    if (!inside)
                    {
                        continue;
                    }

                    switch (stroke.operation)
                    {
                        case BrushOperation::Paint:
                            output->SetPixel(idx, stroke.labelValue);
                            break;

                        case BrushOperation::Erase:
                            output->SetPixel(idx, static_cast<OutputPixelType>(0));
                            break;

                        case BrushOperation::ThresholdPaint:
                        {
                            // Only paint where the underlying intensity lies
                            // inside the configured window.
                            const double value =
                                static_cast<double>(input->GetPixel(idx));
                            if (value >= m_ThresholdLow && value <= m_ThresholdHigh)
                            {
                                output->SetPixel(idx, stroke.labelValue);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

} // namespace medaxis

#endif // medAxisBrushFilter_hxx
