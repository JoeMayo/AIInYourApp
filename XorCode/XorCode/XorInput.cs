using Microsoft.ML.Data;

namespace XorCode
{
    public class XorInput
    {
        [ColumnName("X1"), LoadColumn(0)]
        public float X1 { get; set; }

        [ColumnName("X2"), LoadColumn(1)]
        public float X2 { get; set; }

        [ColumnName("Y"), LoadColumn(2)]
        public bool Y { get; set; }
    }
}
