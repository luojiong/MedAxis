/*=========================================================================

  medAxisAutoFillFilter.hxx

=========================================================================*/
#ifndef medAxisAutoFillFilter_hxx
#define medAxisAutoFillFilter_hxx

#include "medAxisAutoFillFilter.h"

#include "itkImageRegionConstIterator.h"

#include <vector>

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
AutoFillFilter<TInputImage, TOutputImage>::AutoFillFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
void
AutoFillFilter<TInputImage, TOutputImage>::GenerateData()
{
    const auto* input = this->GetInput();
    const auto region = input->GetLargestPossibleRegion();
    const auto& size = region.GetSize();

    this->AllocateOutputs();
    typename TOutputImage::Pointer output = this->GetOutput();

    // Copy the input labels into the output first.
    {
        itk::ImageRegionConstIterator<TInputImage> it(input, region);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            output->SetPixel(it.GetIndex(),
                             static_cast<OutputPixelType>(it.Get()));
        }
    }

    const unsigned int axis = m_SliceAxis;
    const auto numSlices = static_cast<long>(size[axis]);

    // Helper: does a slice contain the target label?
    auto sliceHasLabel = [&](long s) {
        typename TInputImage::IndexType idx{};
        idx[axis] = s;

        // Iterate over all voxels of the slice.
        std::vector<long> bounds;
        for (unsigned int d = 0; d < TInputImage::ImageDimension; ++d)
        {
            if (d != axis) bounds.push_back(static_cast<long>(size[d]));
        }

        const long nx = bounds.size() > 0 ? bounds[0] : 1;
        const long ny = bounds.size() > 1 ? bounds[1] : 1;

        for (long y = 0; y < ny; ++y)
        {
            for (long x = 0; x < nx; ++x)
            {
                unsigned int comp = 0;
                for (unsigned int d = 0; d < TInputImage::ImageDimension; ++d)
                {
                    if (d == axis) continue;
                    idx[d] = (comp == 0) ? x : y;
                    ++comp;
                }
                if (static_cast<OutputPixelType>(input->GetPixel(idx)) == m_LabelValue)
                {
                    return true;
                }
            }
        }
        return false;
    };

    // Find labeled slices.
    std::vector<long> labeled;
    labeled.reserve(numSlices);
    for (long s = 0; s < numSlices; ++s)
    {
        if (sliceHasLabel(s))
        {
            labeled.push_back(s);
        }
    }

    if (labeled.size() < 2)
    {
        return; // nothing to interpolate
    }

    // NOTE(Phase 1): Morphological / BSpline modes currently share the
    // linear implementation; replace with morphological interpolation and
    // B-spline contour fitting respectively.
    auto fillSlice = [&](long z, long s0, long s1) {
        const double t = static_cast<double>(z - s0) / static_cast<double>(s1 - s0);

        std::vector<long> bounds;
        for (unsigned int d = 0; d < TInputImage::ImageDimension; ++d)
        {
            if (d != axis) bounds.push_back(static_cast<long>(size[d]));
        }
        const long nx = bounds.size() > 0 ? bounds[0] : 1;
        const long ny = bounds.size() > 1 ? bounds[1] : 1;

        typename TInputImage::IndexType idx{}, idx0{}, idx1{};
        idx[axis] = z;
        idx0[axis] = s0;
        idx1[axis] = s1;

        for (long y = 0; y < ny; ++y)
        {
            for (long x = 0; x < nx; ++x)
            {
                unsigned int comp = 0;
                for (unsigned int d = 0; d < TInputImage::ImageDimension; ++d)
                {
                    if (d == axis) continue;
                    const long v = (comp == 0) ? x : y;
                    idx[d] = idx0[d] = idx1[d] = v;
                    ++comp;
                }

                const double m0 =
                    (static_cast<OutputPixelType>(input->GetPixel(idx0)) == m_LabelValue) ? 1.0 : 0.0;
                const double m1 =
                    (static_cast<OutputPixelType>(input->GetPixel(idx1)) == m_LabelValue) ? 1.0 : 0.0;

                const double blended = m0 * (1.0 - t) + m1 * t;
                output->SetPixel(idx, blended >= 0.5
                                        ? m_LabelValue
                                        : static_cast<OutputPixelType>(0));
            }
        }
    };

    for (size_t i = 0; i + 1 < labeled.size(); ++i)
    {
        const long s0 = labeled[i];
        const long s1 = labeled[i + 1];
        for (long z = s0 + 1; z < s1; ++z)
        {
            fillSlice(z, s0, s1);
        }
    }
}

} // namespace medaxis

#endif // medAxisAutoFillFilter_hxx
