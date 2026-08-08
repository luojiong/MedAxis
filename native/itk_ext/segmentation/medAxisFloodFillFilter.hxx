/*=========================================================================

  medAxisFloodFillFilter.hxx

=========================================================================*/
#ifndef medAxisFloodFillFilter_hxx
#define medAxisFloodFillFilter_hxx

#include "medAxisFloodFillFilter.h"

#include <array>
#include <queue>
#include <vector>

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
FloodFillFilter<TInputImage, TOutputImage>::FloodFillFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
FloodFillFilter<TInputImage, TOutputImage>::GenerateData()
{
    constexpr unsigned int Dim = TInputImage::ImageDimension;

    const auto* input = this->GetInput();
    const auto region = input->GetLargestPossibleRegion();

    if (!region.IsInside(m_Seed))
    {
        itkExceptionMacro("FloodFillFilter: seed is outside the image.");
    }

    const double seedValue = static_cast<double>(input->GetPixel(m_Seed));
    if (seedValue < m_LowerThreshold || seedValue > m_UpperThreshold)
    {
        itkExceptionMacro("FloodFillFilter: seed intensity is outside the threshold window.");
    }

    // Neighbor offsets for 6 / 18 / 26 connectivity (3D).
    static const std::array<std::array<int, 3>, 26> offsets = {{
        // 6 face-connected
        { 1, 0, 0}, {-1, 0, 0}, {0,  1, 0}, {0, -1, 0}, {0, 0,  1}, {0, 0, -1},
        // 12 edge-connected
        { 1, 1, 0}, { 1,-1, 0}, {-1,  1, 0}, {-1,-1, 0},
        { 1, 0, 1}, { 1, 0,-1}, {-1,  0, 1}, {-1, 0,-1},
        { 0, 1, 1}, { 0, 1,-1}, { 0, -1, 1}, { 0,-1,-1},
        // 8 corner-connected
        { 1, 1, 1}, { 1, 1,-1}, { 1,-1, 1}, { 1,-1,-1},
        {-1, 1, 1}, {-1, 1,-1}, {-1,-1, 1}, {-1,-1,-1}
    }};

    int numOffsets = 6;
    if (m_Connectivity >= 18) numOffsets = 18;
    if (m_Connectivity >= 26) numOffsets = 26;

    this->AllocateOutputs();
    typename TOutputImage::Pointer output = this->GetOutput();
    output->FillBuffer(static_cast<OutputPixelType>(0));

    std::queue<IndexType> frontier;
    frontier.push(m_Seed);
    output->SetPixel(m_Seed, m_ReplaceValue);
    m_FilledCount = 1;

    const auto& size = region.GetSize();
    using IndexValueType = typename IndexType::IndexValueType;

    while (!frontier.empty())
    {
        const IndexType current = frontier.front();
        frontier.pop();

        for (int n = 0; n < numOffsets; ++n)
        {
            IndexType neighbor = current;
            bool inside = true;
            for (unsigned int d = 0; d < Dim; ++d)
            {
                const int off = (d < 3) ? offsets[n][d] : 0;
                neighbor[d] = current[d] + static_cast<IndexValueType>(off);
                if (neighbor[d] < 0 ||
                    neighbor[d] >= static_cast<IndexValueType>(size[d]))
                {
                    inside = false;
                    break;
                }
            }
            if (!inside)
            {
                continue;
            }

            // Already filled?
            if (output->GetPixel(neighbor) == m_ReplaceValue)
            {
                continue;
            }

            const double v = static_cast<double>(input->GetPixel(neighbor));
            if (v >= m_LowerThreshold && v <= m_UpperThreshold)
            {
                output->SetPixel(neighbor, m_ReplaceValue);
                frontier.push(neighbor);
                ++m_FilledCount;
            }
        }
    }
}

} // namespace medaxis

#endif // medAxisFloodFillFilter_hxx
