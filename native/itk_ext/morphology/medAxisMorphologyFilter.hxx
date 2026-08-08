/*=========================================================================

  medAxisMorphologyFilter.hxx

=========================================================================*/
#ifndef medAxisMorphologyFilter_hxx
#define medAxisMorphologyFilter_hxx

#include "medAxisMorphologyFilter.h"

#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryOpeningByReconstructionImageFilter.h"
#include "itkBinaryClosingByReconstructionImageFilter.h"
#include "itkBinaryMorphologicalOpeningImageFilter.h"
#include "itkBinaryMorphologicalClosingImageFilter.h"
#include "itkBinaryWhiteTopHatImageFilter.h"
#include "itkBinaryBlackTopHatImageFilter.h"

#include "itkGrayscaleErodeImageFilter.h"
#include "itkGrayscaleDilateImageFilter.h"
#include "itkGrayscaleMorphologicalOpeningImageFilter.h"
#include "itkGrayscaleMorphologicalClosingImageFilter.h"
#include "itkGrayscaleWhiteTopHatImageFilter.h"
#include "itkGrayscaleBlackTopHatImageFilter.h"
#include "itkGradientMorphologicalImageFilter.h"

#include "itkAnnulusStructuringElement.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
MorphologyFilter<TInputImage, TOutputImage>::MorphologyFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
template <typename TFilter, typename TKernel>
void
MorphologyFilter<TInputImage, TOutputImage>::RunFilter(const TKernel& kernel)
{
    typename TFilter::Pointer f = TFilter::New();
    f->SetInput(this->GetInput());
    f->SetKernel(kernel);

    this->AllocateOutputs();
    f->GraftOutput(this->GetOutput());
    f->Update();
    this->GraftOutput(f->GetOutput());
}

template <typename TInputImage, typename TOutputImage>
template <typename TKernel>
void
MorphologyFilter<TInputImage, TOutputImage>::DispatchWithKernel(const TKernel& kernel)
{
    using In = TInputImage;
    using Out = TOutputImage;
    using K = TKernel;

    switch (this->m_Operation)
    {
        case MorphOperation::Erode:
            if (this->m_BinaryMode)
                this->RunFilter<itk::BinaryErodeImageFilter<In, Out, K>>(kernel);
            else
                this->RunFilter<itk::GrayscaleErodeImageFilter<In, Out, K>>(kernel);
            break;

        case MorphOperation::Dilate:
            if (this->m_BinaryMode)
                this->RunFilter<itk::BinaryDilateImageFilter<In, Out, K>>(kernel);
            else
                this->RunFilter<itk::GrayscaleDilateImageFilter<In, Out, K>>(kernel);
            break;

        case MorphOperation::Open:
            if (this->m_BinaryMode)
                this->RunFilter<itk::BinaryMorphologicalOpeningImageFilter<In, Out, K>>(kernel);
            else
                this->RunFilter<itk::GrayscaleMorphologicalOpeningImageFilter<In, Out, K>>(kernel);
            break;

        case MorphOperation::Close:
            if (this->m_BinaryMode)
                this->RunFilter<itk::BinaryMorphologicalClosingImageFilter<In, Out, K>>(kernel);
            else
                this->RunFilter<itk::GrayscaleMorphologicalClosingImageFilter<In, Out, K>>(kernel);
            break;

        case MorphOperation::TopHat:
            if (this->m_BinaryMode)
                this->RunFilter<itk::BinaryWhiteTopHatImageFilter<In, Out, K>>(kernel);
            else
                this->RunFilter<itk::GrayscaleWhiteTopHatImageFilter<In, Out, K>>(kernel);
            break;

        case MorphOperation::BottomHat:
            if (this->m_BinaryMode)
                this->RunFilter<itk::BinaryBlackTopHatImageFilter<In, Out, K>>(kernel);
            else
                this->RunFilter<itk::GrayscaleBlackTopHatImageFilter<In, Out, K>>(kernel);
            break;

        case MorphOperation::Gradient:
            // Gradient morphology is defined for grayscale images only.
            this->RunFilter<itk::GradientMorphologicalImageFilter<In, Out, K>>(kernel);
            break;
    }
}

template <typename TInputImage, typename TOutputImage>
void
MorphologyFilter<TInputImage, TOutputImage>::GenerateData()
{
    constexpr unsigned int Dim = TInputImage::ImageDimension;
    using StructuringElement = itk::FlatStructuringElement<Dim>;

    typename StructuringElement::RadiusType radius;
    radius.Fill(this->m_Radius);

    switch (this->m_KernelType)
    {
        case MorphKernelType::Ball:
        {
            const auto kernel = StructuringElement::Ball(radius);
            this->DispatchWithKernel(kernel);
            break;
        }
        case MorphKernelType::Box:
        {
            const auto kernel = StructuringElement::Box(radius);
            this->DispatchWithKernel(kernel);
            break;
        }
        case MorphKernelType::Cross:
        {
            const auto kernel = StructuringElement::Cross(radius);
            this->DispatchWithKernel(kernel);
            break;
        }
        case MorphKernelType::Annulus:
        {
            using AnnulusKernel = itk::AnnulusStructuringElement<InputPixelType, Dim>;
            AnnulusKernel kernel;
            typename AnnulusKernel::SizeType size;
            size.Fill(this->m_Radius);
            kernel.SetRadius(size);
            kernel.CreateStructuringElement();
            this->DispatchWithKernel(kernel);
            break;
        }
    }
}

} // namespace medaxis

#endif // medAxisMorphologyFilter_hxx
