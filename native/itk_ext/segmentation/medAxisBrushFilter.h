/*=========================================================================

  medAxisBrushFilter.h

  3D brush painting filter. Accumulates strokes (center, shape, operation,
  radius, label value) and applies them all to a label image in a single
  pass, touching only the bounding boxes of the brushes for efficiency.

=========================================================================*/
#ifndef medAxisBrushFilter_h
#define medAxisBrushFilter_h

#include "itkImageToImageFilter.h"
#include "itkIndex.h"

#include <vector>

namespace medaxis
{

enum class BrushShape
{
    Sphere,
    Box,
    Gaussian3D
};

enum class BrushOperation
{
    Paint,
    Erase,
    ThresholdPaint
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT BrushFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(BrushFilter);

    using Self = BrushFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;
    using IndexType = itk::Index<ImageDimension>;
    using OutputPixelType = typename TOutputImage::PixelType;

    struct Stroke
    {
        IndexType       center{};
        BrushShape      shape = BrushShape::Sphere;
        BrushOperation  operation = BrushOperation::Paint;
        double          radius = 3.0;          ///< in voxels
        OutputPixelType labelValue = static_cast<OutputPixelType>(1);
    };

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(BrushFilter);

    /// Stroke accumulation API.
    void AddStroke(const Stroke& stroke) { m_Strokes.push_back(stroke); this->Modified(); }
    void ClearStrokes() { m_Strokes.clear(); this->Modified(); }
    void SetStrokes(const std::vector<Stroke>& strokes) { m_Strokes = strokes; this->Modified(); }
    const std::vector<Stroke>& GetStrokes() const { return m_Strokes; }

    /// Intensity window for ThresholdPaint strokes.
    itkSetMacro(ThresholdLow, double);
    itkGetConstReferenceMacro(ThresholdLow, double);
    itkSetMacro(ThresholdHigh, double);
    itkGetConstReferenceMacro(ThresholdHigh, double);

    /// Gaussian brush cutoff (apply where weight >= threshold).
    itkSetMacro(GaussianCutoff, double);
    itkGetConstReferenceMacro(GaussianCutoff, double);

protected:
    BrushFilter();
    ~BrushFilter() override = default;

    void GenerateData() override;

private:
    std::vector<Stroke> m_Strokes;
    double              m_ThresholdLow = 0.0;
    double              m_ThresholdHigh = 0.0;
    double              m_GaussianCutoff = 0.05;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisBrushFilter.hxx"
#endif

#endif // medAxisBrushFilter_h
