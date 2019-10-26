using Microsoft.ML;
using System;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using Xamarin.Forms;

namespace XorApp
{
    // Learn more about making custom code visible in the Xamarin.Forms previewer
    // by visiting https://aka.ms/xamarinforms-previewer
    [DesignTimeVisible(false)]
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();
        }

        void PredictButton_Clicked(object sender, EventArgs e)
        {
            var xorInput =
                new XorInput
                {
                    X1 = float.Parse(X1.Text),
                    X2 = float.Parse(X2.Text)
                };

            var ctx = new MLContext();

            // Reminder: set Build Action property of Model.zip file to "Embedded Resource"
            var assembly = IntrospectionExtensions.GetTypeInfo(typeof(MainPage)).Assembly;
            Stream stream = assembly.GetManifestResourceStream("XorApp.XorModel.zip");

            ITransformer xorModel = ctx.Model.Load(stream, out var modelInputSchema);

            var predEngine = ctx.Model.CreatePredictionEngine<XorInput, XorOutput>(xorModel);

            XorOutput result = predEngine.Predict(xorInput);

            XorResult.Text = result.Prediction.ToString();
        }
    }
}
