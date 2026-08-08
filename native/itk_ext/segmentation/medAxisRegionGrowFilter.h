/*=========================================================================

  medAxisRegionGrowFilter.h

  Region growing segmentation with three ITK backends:
    Connected    - itk::ConnectedThresholdImageFilter
    Confidence   - itk::ConfidenceConnectedImageFilter
    Neighborhood - itk::NeighborhoodConnectedImageFilter

=========================================================================*/
#ifndef medAxisRegionGrowFilter_h
#define medAxisRegionGrowFilter_h

#include "itkImageToImageFilter.h"
#include "itkIndex.h"

#include <vector>

namespace medaxis
{

enum class RegionGrowMode
{
    Connected,
    Confidence,
    Neighborhood
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT RegionGrowFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(RegionGrowFilter);

    using Self = RegionGrowFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    using IndexType = typename TInputImage::IndexType;
    using OutputPixelType = typename TOutputImage::PixelType;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(RegionGrowFilter);

    itkSetMacro(Mode, RegionGrowMode);
    itkGetConstReferenceMacro(Mode, RegionGrowMode);

    void AddSeed(const IndexType& seed) { m_Seeds.push_back(seed); this->Modified(); }
    void ClearSeeds() { m_Seeds.clear(); this->Modified(); }

    /// Intensity window for Connected / Neighborhood modes.
    itkSetMacro(LowerThreshold, double);
    itkGetConstReferenceMacro(LowerThreshold, double);
    itkSetMacro(UpperThreshold, double);
    itkGetConstReferenceMacro(UpperThreshold, double);

    /// Confidence mode: mean +/- Multiplier * std.
    itkSetMacro(Multiplier, double);
    itkGetConstReferenceMacro(Multiplier, double);

    /// Confidence mode: number of statistic re-estimation iterations.
    itkSetMacro(Iterations, unsigned int);
    itkGetConstReferenceMacro(Iterations, unsigned int);

    /// Neighborhood mode: neighborhood radius.
    itkSetMacro(NeighborhoodRadius, unsigned int);
    itkGetConstReferenceMacro(NeighborhoodRadius, unsigned int);

    itkSetMacro(ReplaceValue, OutputPixelType);
    itkGetConstReferenceMacro(ReplaceValue, OutputPixelType);

protected:
    RegionGrowFilter();
    ~RegionGrowFilter() override = default;

    void GenerateData() override;

private:
    RegionGrowMode          m_Mode = RegionGrowMode::Connected;
    std::vector<IndexType>  m_Seeds;
    double                  m_LowerThreshold = 0.0;
    double                  m_UpperThreshold = 0.0;
    double                  m_Multiplier = 2.5;
    unsigned int            m_Iterations = 3;
    unsigned int            m_NeighborhoodRadius = 1;
    OutputPixelType         m_ReplaceValue = static_cast<OutputPixelType>(1);
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisRegionGrowFilter.hxx"
#endif

#endif // medAxisRegionGrowFilter_h
