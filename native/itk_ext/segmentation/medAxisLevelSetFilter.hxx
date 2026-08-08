/*=========================================================================

  medAxisLevelSetFilter.hxx

=========================================================================*/
#ifndef medAxisLevelSetFilter_hxx
#define medAxisLevelSetFilter_hxx

#include "medAxisLevelSetFilter.h"

#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkSigmoidImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"

#include "itkGeodesicActiveContourLevelSetImageFilter.h"
#include "itkShapeDetectionLevelSetImageFilter.h"
#include "itkThresholdSegmentationLevelSetImageFilter.h"
#include "itkLaplacianSegmentationLevelSetImageFilter.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace medaxis
{

template <typename TInputImage, typename TOutputImage>
LevelSetFilter<TInputImage, TOutputImage>::LevelSetFilter()
{
    this->DynamicMultiThreadingOff();
    m_SeedPoint.Fill(0.0);
}

template <typename TInputImage, typename TOutputImage>
void
LevelSetFilter<TInputImage, TOutputImage>::GenerateData()
{
    using In = TInputImage;
    using Out = TOutputImage;
    constexpr unsigned int Dim = In::ImageDimension;

    using MaskImageType = itk::Image<unsigned char, Dim>;
    using LevelSetImageType = itk::Image<float, Dim>;
    using FeatureImageType = itk::Image<float, Dim>;

    const In* input = this->GetInput();

    // ------------------------------------------------------------------
    // 1. Initial level set: signed distance map of a seed ball.
    // ------------------------------------------------------------------
    typename MaskImageType::Pointer seedMask = MaskImageType::New();
    seedMask->CopyInformation(input);
    seedMask->SetRegions(input->GetLargestPossibleRegion());
    seedMask->Allocate();
    seedMask->FillBuffer(0);

    typename MaskImageType::IndexType seedIndex;
    input->TransformPhysicalPointToIndex(m_SeedPoint, seedIndex);
    if (seedMask->GetLargestPossibleRegion().IsInside(seedIndex))
    {
        seedMask->SetPixel(seedIndex, 1);
    }

    using BallType = itk::BinaryBallStructuringElement<unsigned char, Dim>;
    BallType ball;
    typename BallType::SizeType ballRadius;
    for (unsigned int d = 0; d < Dim; ++d)
    {
        ballRadius[d] = static_cast<typename BallType::SizeValueType>(
            std::max(1.0, m_SeedRadius / input->GetSpacing()[d]));
    }
    ball.SetRadius(ballRadius);
    ball.CreateStructuringElement();

    typename itk::BinaryDilateImageFilter<MaskImageType, MaskImageType, BallType>::Pointer
        dilater = itk::BinaryDilateImageFilter<MaskImageType, MaskImageType, BallType>::New();
    dilater->SetInput(seedMask);
    dilater->SetKernel(ball);
    dilater->SetDilateValue(1);
    dilater->Update();

    typename itk::SignedMaurerDistanceMapImageFilter<MaskImageType, LevelSetImageType>::Pointer
        distance = itk::SignedMaurerDistanceMapImageFilter<MaskImageType, LevelSetImageType>::New();
    distance->SetInput(dilater->GetOutput());
    distance->SetInsideIsPositive(false); // negative inside the seed
    distance->SetSquaredDistance(false);
    distance->SetUseImageSpacing(true);
    distance->Update();

    typename LevelSetImageType::Pointer initialLevelSet = distance->GetOutput();
    initialLevelSet->DisconnectPipeline();

    // ------------------------------------------------------------------
    // 2. Feature image (edge indicator) for Geodesic / ShapeDetection.
    // ------------------------------------------------------------------
    typename itk::GradientMagnitudeRecursiveGaussianImageFilter<In, FeatureImageType>::Pointer
        gradient = itk::GradientMagnitudeRecursiveGaussianImageFilter<In, FeatureImageType>::New();
    gradient->SetInput(input);
    gradient->SetSigma(1.0);
    gradient->Update();

    typename itk::StatisticsImageFilter<FeatureImageType>::Pointer stats =
        itk::StatisticsImageFilter<FeatureImageType>::New();
    stats->SetInput(gradient->GetOutput());
    stats->Update();
    const float gmin = stats->GetMinimum();
    const float gmax = std::max(gmin + 1.0e-6f, stats->GetMaximum());

    typename itk::SigmoidImageFilter<FeatureImageType, FeatureImageType>::Pointer sigmoid =
        itk::SigmoidImageFilter<FeatureImageType, FeatureImageType>::New();
    sigmoid->SetInput(gradient->GetOutput());
    sigmoid->SetOutputMinimum(0.0f);
    sigmoid->SetOutputMaximum(1.0f);
    sigmoid->SetAlpha(0.5f * (gmin + gmax));   // inflection point
    sigmoid->SetBeta((gmax - gmin) / 6.0f);    // transition width
    sigmoid->Update();

    typename FeatureImageType::Pointer feature = sigmoid->GetOutput();
    feature->DisconnectPipeline();

    // ------------------------------------------------------------------
    // 3. Run the selected level-set evolution, then binarize.
    // ------------------------------------------------------------------
    typename LevelSetImageType::Pointer evolved;

    switch (m_Mode)
    {
        case LevelSetMode::Geodesic:
        {
            auto ls = itk::GeodesicActiveContourLevelSetImageFilter<
                LevelSetImageType, FeatureImageType>::New();
            ls->SetInput(initialLevelSet);
            ls->SetFeatureImage(feature);
            ls->SetPropagationScaling(static_cast<float>(m_PropagationScaling));
            ls->SetCurvatureScaling(static_cast<float>(m_CurvatureScaling));
            ls->SetAdvectionScaling(static_cast<float>(m_AdvectionScaling));
            ls->SetMaximumRMSError(0.02f);
            ls->SetNumberOfIterations(m_MaxIterations);
            ls->Update();
            evolved = ls->GetOutput();
            evolved->DisconnectPipeline();
            break;
        }

        case LevelSetMode::ShapeDetection:
        {
            auto ls = itk::ShapeDetectionLevelSetImageFilter<
                LevelSetImageType, FeatureImageType>::New();
            ls->SetInput(initialLevelSet);
            ls->SetFeatureImage(feature);
            ls->SetPropagationScaling(static_cast<float>(m_PropagationScaling));
            ls->SetCurvatureScaling(static_cast<float>(m_CurvatureScaling));
            ls->SetEdgeWeight(1.0f);
            ls->SetMaximumRMSError(0.02f);
            ls->SetNumberOfIterations(m_MaxIterations);
            ls->Update();
            evolved = ls->GetOutput();
            evolved->DisconnectPipeline();
            break;
        }

        case LevelSetMode::Threshold:
        {
            using ThresholdLS = itk::ThresholdSegmentationLevelSetImageFilter<
                LevelSetImageType, In>;
            typename ThresholdLS::Pointer ls = ThresholdLS::New();
            ls->SetInput(initialLevelSet);
            ls->SetFeatureImage(const_cast<In*>(input));
            ls->SetLowerThreshold(static_cast<float>(m_LowerThreshold));
            ls->SetUpperThreshold(static_cast<float>(m_UpperThreshold));
            ls->SetCurvatureScaling(static_cast<float>(m_CurvatureScaling));
            ls->SetEdgeWeight(1.0f);
            ls->SetMaximumRMSError(0.02f);
            ls->SetNumberOfIterations(m_MaxIterations);
            ls->Update();
            evolved = ls->GetOutput();
            evolved->DisconnectPipeline();
            break;
        }

        case LevelSetMode::Laplacian:
        {
            using LaplacianLS = itk::LaplacianSegmentationLevelSetImageFilter<
                LevelSetImageType, In>;
            typename LaplacianLS::Pointer ls = LaplacianLS::New();
            ls->SetInput(initialLevelSet);
            ls->SetFeatureImage(const_cast<In*>(input));
            ls->SetFeatureScaling(static_cast<float>(m_PropagationScaling));
            ls->SetNumberOfIterations(m_MaxIterations);
            ls->SetMaximumRMSError(0.02f);
            ls->Update();
            evolved = ls->GetOutput();
            evolved->DisconnectPipeline();
            break;
        }
    }

    // ------------------------------------------------------------------
    // 4. Binarize: level set < 0 is inside the segmentation.
    // ------------------------------------------------------------------
    typename itk::BinaryThresholdImageFilter<LevelSetImageType, Out>::Pointer binarizer =
        itk::BinaryThresholdImageFilter<LevelSetImageType, Out>::New();
    binarizer->SetInput(evolved);
    binarizer->SetLowerThreshold(std::numeric_limits<float>::lowest());
    binarizer->SetUpperThreshold(0.0f);
    binarizer->SetInsideValue(m_OutputValue);
    binarizer->SetOutsideValue(static_cast<typename Out::PixelType>(0));

    this->AllocateOutputs();
    binarizer->GraftOutput(this->GetOutput());
    binarizer->Update();
    this->GraftOutput(binarizer->GetOutput());
}

} // namespace medaxis

#endif // medAxisLevelSetFilter_hxx
