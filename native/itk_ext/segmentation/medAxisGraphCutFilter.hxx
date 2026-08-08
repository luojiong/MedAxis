/*=========================================================================

  medAxisGraphCutFilter.hxx

  Phase 0 skeleton implementation.

  Current behaviour: paints foreground seed neighborhoods into the output.
  TODO(Phase 1): replace with Boykov-Kolmogorov max-flow:
    - n-links  : exp(-(I_i - I_j)^2 / (2 beta)) weighted by Lambda
    - t-links  : regional probabilities from fg/bg intensity models
                 (RegionSigma) with hard constraints at seeds.

=========================================================================*/
#ifndef medAxisGraphCutFilter_hxx
#define medAxisGraphCutFilter_hxx

#include "medAxisGraphCutFilter.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include <queue>

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
GraphCutFilter<TInputImage, TOutputImage>::GraphCutFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
GraphCutFilter<TInputImage, TOutputImage>::GenerateData()
{
    if (m_FgSeeds.empty())
    {
        itkExceptionMacro("GraphCutFilter: no foreground seeds provided.");
    }

    this->AllocateOutputs();
    typename TOutputImage::Pointer output = this->GetOutput();
    output->FillBuffer(m_BackgroundValue);

    const auto region = output->GetLargestPossibleRegion();
    const int r = static_cast<int>(m_SeedRadius);

    // Provisional skeleton: dilate each foreground seed into a ball.
    for (const auto& seed : m_FgSeeds)
    {
        typename TOutputImage::IndexType idx;
        const auto size = region.GetSize();

        for (int dz = -r; dz <= r; ++dz)
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx)
                {
                    if (dx * dx + dy * dy + dz * dz > r * r)
                    {
                        continue;
                    }
                    idx[0] = seed[0] + dx;
                    idx[1] = seed[1] + dy;
                    idx[2] = (TOutputImage::ImageDimension > 2) ? seed[2] + dz : seed[2];

                    bool inside = true;
                    for (unsigned int d = 0; d < TOutputImage::ImageDimension; ++d)
                    {
                        if (idx[d] < 0 || idx[d] >= static_cast<typename TOutputImage::IndexType::IndexValueType>(size[d]))
                        {
                            inside = false;
                            break;
                        }
                    }
                    if (inside)
                    {
                        output->SetPixel(idx, m_ForegroundValue);
                    }
                }
    }
}

} // namespace medaxis

#endif // medAxisGraphCutFilter_hxx
