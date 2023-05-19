use std::{fs::File, path::Path};

use polars::{
    lazy::dsl::{col, concat_list},
    prelude::{
        CsvWriter, DataFrame, Int64Chunked, IntoLazy, LazyCsvReader, LazyFileListReader, NamedFrom,
        PolarsError, PolarsResult, SerWriter,
    },
    series::{IntoSeries, Series},
};
use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::basic::matrix::DenseMatrix,
};

fn read_csv(path: impl AsRef<Path>) -> PolarsResult<DataFrame> {
    let df = LazyCsvReader::new(path)
        .has_header(true)
        .finish()?
        .collect()?;
    Ok(df)
}

fn write_csv(path: impl AsRef<Path>, df: &mut DataFrame) -> PolarsResult<()> {
    let mut f = File::create(path).unwrap();

    CsvWriter::new(&mut f)
        .has_header(true)
        .with_delimiter(b',')
        .finish(df)?;
    Ok(())
}

struct DataSet {
    target: Vec<i64>,
    data: Vec<Vec<f64>>,
    _df: DataFrame,
}

impl DataSet {
    fn new(df: &mut DataFrame) -> Self {
        let df = df
            .apply("Sex", |s: &Series| {
                s.utf8()
                    .unwrap()
                    .into_no_null_iter()
                    .map(|s| if s == "female" { Some(1) } else { Some(0) })
                    .collect::<Int64Chunked>()
                    .into_series()
            })
            .unwrap();

        let survived = df
            .column("Survived")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<i64>>();

        let n = [
            concat_list([col("Pclass"), col("Sex"), col("SibSp"), col("Parch")])
                .unwrap()
                .alias("data"),
        ];
        let df = df.clone().lazy().select(n).collect().unwrap();

        let data = df
            .column("data")
            .unwrap()
            .list()
            .unwrap()
            .into_no_null_iter()
            .map(|n| {
                n.i64()
                    .unwrap()
                    .into_no_null_iter()
                    .map(|n| n as f64)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>();

        DataSet {
            target: survived,
            data,
            _df: df,
        }
    }

    fn to_matrix(&self) -> DenseMatrix<f64> {
        DenseMatrix::from_2d_vec(&self.data)
    }
}

fn main() -> Result<(), PolarsError> {
    // read the traning set and create a data_set from the givven data
    let mut df = read_csv("data/train.csv")?;
    let data_set = DataSet::new(&mut df);

    // configure the random forest and train the classifier
    let params = RandomForestClassifierParameters::default()
        .with_n_trees(500)
        .with_m(1)
        .with_max_depth(5);
    let classifier =
        RandomForestClassifier::fit(&data_set.to_matrix(), &data_set.target, params).unwrap();

    // read the actual data and create a data_set
    let mut df = read_csv("data/test.csv")?;
    let data_set = DataSet::new(&mut df);

    // run the prediction with the actual data
    let result = classifier.predict(&data_set.to_matrix()).unwrap();

    // create a new dataframe with the result
    let n = 892 + result.len() as u32;
    let p = Series::new("PassangerId", 892..n);
    let s = Series::new("Survived", result);
    let mut df = DataFrame::new(vec![p, s]).unwrap();

    // write the result to a csv file
    write_csv("submission.csv", &mut df)?;

    // load the csv and print the result
    let df_csv = read_csv("submission.csv")?;
    println!("{}", df_csv);
    Ok(())
}
