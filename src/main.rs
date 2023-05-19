use std::fs::File;

use polars::{
    lazy::dsl::{col, concat_list},
    prelude::{
        CsvReader, CsvWriter, DataFrame, Int64Chunked, IntoLazy, LazyCsvReader, LazyFileListReader,
        NamedFrom, PolarsError, SerReader, SerWriter,
    },
    series::{IntoSeries, Series},
};
use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::basic::matrix::DenseMatrix,
};

fn main() -> Result<(), PolarsError> {
    let mut df = LazyCsvReader::new("data/train.csv")
        .has_header(true)
        .finish()?
        .collect()?;

    let df = df.apply("Sex", |s: &Series| {
        s.utf8()
            .unwrap()
            .into_no_null_iter()
            .map(|s| if s == "female" { Some(1) } else { Some(0) })
            .collect::<Int64Chunked>()
            .into_series()
    })?;

    let survived = df
        .column("Survived")?
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect::<Vec<i64>>();

    let n = [concat_list([col("Pclass"), col("Sex"), col("SibSp"), col("Parch")])?.alias("data")];
    let df = df.clone().lazy().select(n).collect().unwrap();

    let data = df
        .column("data")?
        .list()
        .unwrap()
        .into_iter()
        .map(|n| {
            n.unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|n| n as f64)
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<Vec<f64>>>();

    let matrix = DenseMatrix::from_2d_vec(&data);

    let params = RandomForestClassifierParameters::default()
        .with_n_trees(500)
        .with_m(1)
        .with_max_depth(5);
    let classifier = RandomForestClassifier::fit(&matrix, &survived, params).unwrap();

    let mut df = LazyCsvReader::new("data/test.csv")
        .has_header(true)
        .finish()
        .unwrap()
        .collect()
        .unwrap();

    let df = df.apply("Sex", |s: &Series| {
        s.utf8()
            .unwrap()
            .into_no_null_iter()
            .map(|s| if s == "female" { Some(1) } else { Some(0) })
            .collect::<Int64Chunked>()
            .into_series()
    })?;

    let n = [concat_list([col("Pclass"), col("Sex"), col("SibSp"), col("Parch")])?.alias("data")];
    let df = df.clone().lazy().select(n).collect().unwrap();

    let data = df
        .column("data")?
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

    let matrix = DenseMatrix::from_2d_vec(&data);
    let y_hat = classifier.predict(&matrix).unwrap();

    let n = 892 + y_hat.len() as u32;
    let p = Series::new("PassangerId", 892..n);
    let s = Series::new("Survived", y_hat);
    let mut df = DataFrame::new(vec![p, s]).unwrap();

    let mut f = File::create("submission.csv").unwrap();
    CsvWriter::new(&mut f)
        .has_header(true)
        .with_delimiter(b',')
        .finish(&mut df)?;

    let df_csv = CsvReader::from_path("submission.csv")
        .unwrap()
        .has_header(true)
        .finish()?;
    println!("{}", df_csv);
    Ok(())
}
