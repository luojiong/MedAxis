/*=========================================================================

  medAxisMorphologyFilter.h

  Unified morphology filter dispatching to the appropriate ITK binary or
  grayscale morphology filter based on Operation / KernelType / mode.

=========================================================================*/
#ifndef medAxisMorphologyFilter_h
#define medAxisMorphologyFilter_h

#include "itkImageToImageFilter.h"
#include "itkFlatStructuringElement.h"

namespace medaxis
{

enum class MorphOperation
{
    Erode,
    Dilate,
    Open,
    Close,
    TopHat,
    BottomHat,
    Gradient
};

enum class MorphKernelType
{
    Ball,
    Box,
    Cross,
    Annulus
};

template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT MorphologyFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(MorphologyFilter);

    using Self = MorphologyFilter;
    using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(MorphologyFilter);

    static constexpr unsigned int ImageDimension = TInputImage::ImageDimension;

    using InputPixelType = typename TInputImage::PixelType;
    using OutputPixelType = typename TOutputImage::PixelType;

    itkSetMacro(Operation, MorphOperation);
    itkGetConstReferenceMacro(Operation, MorphOperation);

    itkSetMacro(KernelType, MorphKernelType);
    itkGetConstReferenceMacro(KernelType, MorphKernelType);

    /// Structuring element radius in voxels.
    itkSetMacro(Radius, unsigned int);
    itkGetConstReferenceMacro(Radius, unsigned int);

    /// When true, binary morphology filters are used.
    itkSetMacro(BinaryMode, bool);
    itkGetConstReferenceMacro(BinaryMode, bool);
    itkBooleanMacro(BinaryMode);

    /// Foreground / erosion value used in binary mode.
    itkSetMacro(ForegroundValue, InputPixelType);
    itkGetConstReferenceMacro(ForegroundValue, InputPixelType);

protected:
    MorphologyFilter();
    ~MorphologyFilter() override = default;

    void GenerateData() override;

    template <typename TKernel>
    void DispatchWithKernel(const TKernel& kernel);

    template <typename TFilter, typename TKernel>
    void RunFilter(const TKernel& kernel);

private:
    MorphOperation   m_Operation = MorphOperation::Dilate;
    MorphKernelType  m_KernelType = MorphKernelType::Ball;
    unsigned int     m_Radius = 1;
    bool             m_BinaryMode = true;
    InputPixelType   m_ForegroundValue = static_cast<InputPixelType>(1);
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisMorphologyFilter.hxx"
#endif

#endif // medAxisMorphologyFilter_h
