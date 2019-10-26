using Microsoft.ML;
using System;

namespace XorCode
{
    class Program
    {
        static void Main(string[] args)
        {
            const string TrainingDataFilePath = @"..\..\..\xor_data.csv";
            const string ModelFilePath = @"..\..\..\XorModel.zip";

            //Step 1. Create a ML Context
            var ctx = new MLContext();

            //Step 2. Read in the input data for model training
            IDataView xorDataView = ctx.Data.LoadFromTextFile<XorInput>(
                                path: TrainingDataFilePath,
                                hasHeader: true,
                                separatorChar: ',',
                                allowQuoting: true,
                                allowSparse: false);

            //Step 3. Build training pipeline
            var dataProcessPipeline = ctx.Transforms.Concatenate("Features", new[] { "X1", "X2" });
            var trainer = ctx.BinaryClassification.Trainers.FastTree(labelColumnName: "Y", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            //Step 4. Train your model
            ITransformer trainedModel = trainingPipeline.Fit(xorDataView);

            //Step 5. Make predictions using your model
            var predictionEngine = ctx.Model.CreatePredictionEngine<XorInput, XorOutput>(trainedModel);

            Console.WriteLine("\nTesting XOR Model\n");
            var bothFalse = 
                new XorInput 
                { 
                    X1 = 0.3237f,
                    X2 = 0.1491f
                };

            var bothFalsePrediction = predictionEngine.Predict(bothFalse);

            Console.WriteLine($"Both False: {bothFalsePrediction.Prediction}");

            var falseAndTrue =
                new XorInput
                {
                    X1 = 0.3237f,
                    X2 = 0.7491f
                };

            var falseAndTruePrediction = predictionEngine.Predict(falseAndTrue);

            Console.WriteLine($"False and True: {falseAndTruePrediction.Prediction}");

            var trueAndFalse =
                new XorInput
                {
                    X1 = 0.9237f,
                    X2 = 0.1491f
                };

            var trueAndFalsePrediction = predictionEngine.Predict(trueAndFalse);

            Console.WriteLine($"True and False: {trueAndFalsePrediction.Prediction}");

            var bothTrue =
                new XorInput
                {
                    X1 = 0.5237f,
                    X2 = 0.8491f
                };

            var bothTruePrediction = predictionEngine.Predict(bothTrue);

            Console.WriteLine($"Both True: {bothTruePrediction.Prediction}");

            //Step 6. Save the Model
            ctx.Model.Save(trainedModel, xorDataView.Schema, ModelFilePath);
        }
    }
}
