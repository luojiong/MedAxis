/*=========================================================================

  medAxisFloodFillFilter.h

  3D flood fill (BFS) from a seed point, constrained to an intensity
  window, with configurable 6 / 18 / 26 connectivity.

=========================================================================*/
#ifndef medAxisFloodFillFilter_h
#define medAxisFloodFillFilter_h

#include "itkImageToImageFilter.h"
#include "itkIndex.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT FloodFillFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(FloodFillFilter);

    using Self = FloodFillFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    using IndexType = typename TInputImage::IndexType;
    using OutputPixelType = typename TOutputImage::PixelType;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(FloodFillFilter);

    itkSetMacro(Seed, IndexType);
    itkGetConstReferenceMacro(Seed, IndexType);

    /// 6 (face), 18 (face+edge) or 26 (full) connectivity.
    itkSetClampMacro(Connectivity, int, 6, 26);
    itkGetConstReferenceMacro(Connectivity, int);

    itkSetMacro(LowerThreshold, double);
    itkGetConstReferenceMacro(LowerThreshold, double);
    itkSetMacro(UpperThreshold, double);
    itkGetConstReferenceMacro(UpperThreshold, double);

    itkSetMacro(ReplaceValue, OutputPixelType);
    itkGetConstReferenceMacro(ReplaceValue, OutputPixelType);

    /// Number of voxels filled by the last run.
    itkGetConstReferenceMacro(FilledCount, size_t);

protected:
    FloodFillFilter();
    ~FloodFillFilter() override = default;

    void GenerateData() override;

private:
    IndexType       m_Seed{};
    int             m_Connectivity = 26;
    double          m_LowerThreshold = 0.0;
    double          m_UpperThreshold = 0.0;
    OutputPixelType m_ReplaceValue = static_cast<OutputPixelType>(1);
    size_t          m_FilledCount = 0;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisFloodFillFilter.hxx"
#endif

#endif // medAxisFloodFillFilter_h
