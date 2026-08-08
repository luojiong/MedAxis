/*=========================================================================

  medAxisDemonsRegistration.h

  Demons-based deformable registration with three variants:
    Classic            - itk::DemonsRegistrationFilter
    Diffeomorphic      - itk::DiffeomorphicDemonsRegistrationFilter
    FastSymmetricForces- itk::FastSymmetricForcesDemonsRegistrationFilter

  Input 0 = fixed, input 1 = moving.
  Output = displacement field (itk::Image<itk::Vector<float,Dim>>).

=========================================================================*/
#ifndef medAxisDemonsRegistration_h
#define medAxisDemonsRegistration_h

#include "itkProcessObject.h"
#include "itkVector.h"
#include "itkImage.h"

namespace medaxis
{

enum class DemonsVariant
{
    Classic,
    Diffeomorphic,
    FastSymmetricForces
};

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT DemonsRegistration : public itk::ProcessObject
{
public:
    ITK_DISALLOW_COPY_AND_MOVE(DemonsRegistration);

    using Self = DemonsRegistration;
    using Superclass = itk::ProcessObject;
    using Pointer = itk::SmartPointer<Self>;
    using ConstPointer = itk::SmartPointer<const Self>;

    static constexpr unsigned int Dimension = TFixedImage::ImageDimension;

    using VectorType = itk::Vector<float, Dimension>;
    using OutputFieldType = itk::Image<VectorType, Dimension>;

    itkNewMacro(Self);
    itkOverrideGetNameOfClassMacro(DemonsRegistration);

    void SetFixedImage(const TFixedImage* image);
    const TFixedImage* GetFixedImage() const;

    void SetMovingImage(const TMovingImage* image);
    const TMovingImage* GetMovingImage() const;

    itkSetMacro(Variant, DemonsVariant);
    itkGetConstReferenceMacro(Variant, DemonsVariant);

    itkSetMacro(NumberOfIterations, unsigned int);
    itkGetConstReferenceMacro(NumberOfIterations, unsigned int);

    itkSetMacro(StandardDeviation, double);
    itkGetConstReferenceMacro(StandardDeviation, double);

    itkSetMacro(MaximumUpdateStepLength, double);
    itkGetConstReferenceMacro(MaximumUpdateStepLength, double);

    itkSetMacro(SmoothDisplacementField, bool);
    itkGetConstReferenceMacro(SmoothDisplacementField, bool);
    itkBooleanMacro(SmoothDisplacementField);

    OutputFieldType* GetOutput();

protected:
    DemonsRegistration();
    ~DemonsRegistration() override = default;

    void GenerateData() override;

    using Superclass::MakeOutput;
    DataObjectPointer MakeOutput(DataObjectPointerArraySizeType idx) override;

private:
    DemonsVariant m_Variant = DemonsVariant::FastSymmetricForces;
    unsigned int  m_NumberOfIterations = 100;
    double        m_StandardDeviation = 1.0;
    double        m_MaximumUpdateStepLength = 2.0;
    bool          m_SmoothDisplacementField = true;
};

} // namespace medaxis

#ifndef ITK_MANUAL_INSTANTIATION
#  include "medAxisDemonsRegistration.hxx"
#endif

#endif // medAxisDemonsRegistration_h
