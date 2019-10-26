using LinqToTwitter;
using Microsoft.ML;
using System;
using System.ComponentModel;
using System.IO;
using System.Reflection;
using Xamarin.Forms;

namespace TweetSentimentApp
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

        async void TweetButton_Clicked(object sender, EventArgs e)
        {
            bool sendTweet = true;
            string tweetText = TweetEditor.Text;

            bool isPositive = CheckSentiment(tweetText);

            if (!isPositive)
                sendTweet = await DisplayAlert("Are you sure?", "That sounds a bit negative.", "Tweet Anyway", "Cancel");

            if (sendTweet)
            {
                await SendTweetAsync(tweetText);

                await DisplayAlert("Success!", "Tweet Sent.", "OK");
            }
            else
            {
                await DisplayAlert("No Action", "Tweet Not Sent.", "OK");
            }
        }

        async System.Threading.Tasks.Task SendTweetAsync(string tweetText)
        {
            //var auth = new SingleUserAuthorizer
            //{
            //    CredentialStore = new SingleUserInMemoryCredentialStore
            //    {
            //        ConsumerKey = "ConsumerKey",
            //        ConsumerSecret = "ConsumerSecret",
            //        AccessToken = "AccessToken",
            //        AccessTokenSecret = "AccessTokenSecret"
            //    }
            //};

            //var twitterCtx = new TwitterContext(auth);

            //await twitterCtx.TweetAsync(tweetText);
        }

        bool CheckSentiment(string text)
        {
            var mlContext = new MLContext();

            var assembly = IntrospectionExtensions.GetTypeInfo(typeof(MainPage)).Assembly;
            Stream stream = assembly.GetManifestResourceStream("TweetSentimentApp.MLModel.zip");
            ITransformer mlModel = mlContext.Model.Load(stream, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            var inputTweet = new ModelInput
            {
                Tweet = text
            };

            ModelOutput result = predEngine.Predict(inputTweet);

            return result.Prediction;
        }
    }
}
