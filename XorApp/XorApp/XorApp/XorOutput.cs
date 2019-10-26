using Microsoft.ML.Data;

namespace XorApp
{
    public class XorOutput
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
}
