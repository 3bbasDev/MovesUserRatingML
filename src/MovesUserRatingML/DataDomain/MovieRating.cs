using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MovesUserRatingML.DataDomain
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }
}
