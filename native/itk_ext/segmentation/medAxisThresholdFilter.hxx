/*=========================================================================

  medAxisThresholdFilter.hxx

=========================================================================*/
#ifndef medAxisThresholdFilter_hxx
#define medAxisThresholdFilter_hxx

#include "medAxisThresholdFilter.h"

#include "itkOtsuThresholdImageFilter.h"
#include "itkIsoDataThresholdImageFilter.h"
#include "itkMaximumEntropyThresholdImageFilter.h"
#include "itkMomentsThresholdImageFilter.h"
#include "itkRenyiEntropyThresholdImageFilter.h"
#include "itkHuangThresholdImageFilter.h"
#include "itkIntermodesThresholdImageFilter.h"
#include "itkLiThresholdImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
// Local-mean adaptive threshold (no itkAdaptiveThresholdImageFilter in ITK 5.4)
#include "itkMeanImageFilter.h"
#include "itkSubtractImageFilter.h"

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
ThresholdFilter<TInputImage, TOutputImage>::ThresholdFilter()
{
    this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
template <typename TFilter>
void
ThresholdFilter<TInputImage, TOutputImage>::RunHistogramFilter()
{
    typename TFilter::Pointer f = TFilter::New();
    f->SetInput(this->GetInput());
    f->SetNumberOfHistogramBins(this->m_NumberOfHistogramBins);
    f->SetInsideValue(this->m_InsideValue);
    f->SetOutsideValue(this->m_OutsideValue);

    this->AllocateOutputs();
    f->GraftOutput(this->GetOutput());
    f->Update();
    this->GraftOutput(f->GetOutput());

    this->m_ComputedThreshold = static_cast<double>(f->GetThreshold());
}

template <typename TInputImage, typename TOutputImage>
void
ThresholdFilter<TInputImage, TOutputImage>::GenerateData()
{
    using In = TInputImage;
    using Out = TOutputImage;

    switch (this->m_Method)
    {
        case ThresholdMethod::Otsu:
            this->RunHistogramFilter<itk::OtsuThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::IsoData:
            this->RunHistogramFilter<itk::IsoDataThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::MaxEntropy:
            this->RunHistogramFilter<itk::MaximumEntropyThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::Moments:
            this->RunHistogramFilter<itk::MomentsThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::RenyiEntropy:
            this->RunHistogramFilter<itk::RenyiEntropyThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::Huang:
            this->RunHistogramFilter<itk::HuangThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::Intermodes:
            this->RunHistogramFilter<itk::IntermodesThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::Li:
            this->RunHistogramFilter<itk::LiThresholdImageFilter<In, Out>>();
            break;

        case ThresholdMethod::Adaptive:
        {
            // Local-mean (Bradley) adaptive thresholding. ITK 5.4 has no
            // built-in adaptive threshold filter, so the threshold surface is
            // the local mean over the adaptive radius:
            //   out = (img - local_mean) > 0
            using MeanFilter = itk::MeanImageFilter<TInputImage, TInputImage>;
            typename MeanFilter::Pointer mean = MeanFilter::New();
            typename MeanFilter::RadiusType radius;
            radius.Fill(this->m_AdaptiveRadius);
            mean->SetRadius(radius);
            mean->SetInput(this->GetInput());

            using SubtractFilter = itk::SubtractImageFilter<TInputImage, TInputImage, TInputImage>;
            typename SubtractFilter::Pointer sub = SubtractFilter::New();
            sub->SetInput1(this->GetInput());
            sub->SetInput2(mean->GetOutput());

            using BinaryFilter = itk::BinaryThresholdImageFilter<TInputImage, TOutputImage>;
            typename BinaryFilter::Pointer f = BinaryFilter::New();
            f->SetInput(sub->GetOutput());
            f->SetLowerThreshold(0.0);
            f->SetUpperThreshold(itk::NumericTraits<typename TInputImage::PixelType>::max());
            f->SetInsideValue(this->m_InsideValue);
            f->SetOutsideValue(this->m_OutsideValue);

            this->AllocateOutputs();
            f->GraftOutput(this->GetOutput());
            f->Update();
            this->GraftOutput(f->GetOutput());
            this->m_ComputedThreshold = 0.0; // no global threshold for adaptive
            break;
        }

        case ThresholdMethod::Manual:
        {
            typename itk::BinaryThresholdImageFilter<In, Out>::Pointer f =
                itk::BinaryThresholdImageFilter<In, Out>::New();
            f->SetInput(this->GetInput());
            f->SetLowerThreshold(static_cast<typename In::PixelType>(this->m_LowerThreshold));
            f->SetUpperThreshold(static_cast<typename In::PixelType>(this->m_UpperThreshold));
            f->SetInsideValue(this->m_InsideValue);
            f->SetOutsideValue(this->m_OutsideValue);

            this->AllocateOutputs();
            f->GraftOutput(this->GetOutput());
            f->Update();
            this->GraftOutput(f->GetOutput());
            this->m_ComputedThreshold = 0.5 * (this->m_LowerThreshold + this->m_UpperThreshold);
            break;
        }
    }
}

} // namespace medaxis

#endif // medAxisThresholdFilter_hxx
