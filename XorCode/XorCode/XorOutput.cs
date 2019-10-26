using Microsoft.ML.Data;

namespace XorCode
{
    public class XorOutput
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
}
