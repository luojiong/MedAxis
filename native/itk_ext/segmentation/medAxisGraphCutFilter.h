/*=========================================================================

  medAxisGraphCutFilter.h

  Interactive graph-cut segmentation with foreground / background seeds.

  Phase 0 skeleton: stores seeds, weights and lambda. GenerateData()
  produces a provisional seed-dilation result. The max-flow backend
  (Boykov-Kolmogorov via Boost Graph Library / CGAL graph_cut) will be
  wired in Phase 1 without changing this public interface.

=========================================================================*/
#ifndef medAxisGraphCutFilter_h
#define medAxisGraphCutFilter_h

#include "itkImageToImageFilter.h"
#include "itkIndex.h"

#include <vector>

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT GraphCutFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(GraphCutFilter);

    using Self = GraphCutFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;
    using IndexType = typename TInputImage::IndexType;
    using OutputPixelType = typename TOutputImage::PixelType;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(GraphCutFilter);

    /// Seed management.
    void AddForegroundSeed(const IndexType& seed) { m_FgSeeds.push_back(seed); this->Modified(); }
    void AddBackgroundSeed(const IndexType& seed) { m_BgSeeds.push_back(seed); this->Modified(); }
    void ClearSeeds()
    {
        m_FgSeeds.clear();
        m_BgSeeds.clear();
        this->Modified();
    }

    /// Boundary smoothness weight (larger = smoother boundaries).
    itkSetMacro(Lambda, double);
    itkGetConstReferenceMacro(Lambda, double);

    /// Sigma of the intensity model used for the regional term.
    itkSetMacro(RegionSigma, double);
    itkGetConstReferenceMacro(RegionSigma, double);

    itkSetMacro(ForegroundValue, OutputPixelType);
    itkGetConstReferenceMacro(ForegroundValue, OutputPixelType);
    itkSetMacro(BackgroundValue, OutputPixelType);
    itkGetConstReferenceMacro(BackgroundValue, OutputPixelType);

    /// Radius (voxels) of the provisional seed dilation in the skeleton.
    itkSetMacro(SeedRadius, unsigned int);
    itkGetConstReferenceMacro(SeedRadius, unsigned int);

protected:
    GraphCutFilter();
    ~GraphCutFilter() override = default;

    void GenerateData() override;

private:
    std::vector<IndexType> m_FgSeeds;
    std::vector<IndexType> m_BgSeeds;
    double                 m_Lambda = 50.0;
    double                 m_RegionSigma = 10.0;
    OutputPixelType        m_ForegroundValue = static_cast<OutputPixelType>(1);
    OutputPixelType        m_BackgroundValue = static_cast<OutputPixelType>(0);
    unsigned int           m_SeedRadius = 2;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisGraphCutFilter.hxx"
#endif

#endif // medAxisGraphCutFilter_h
