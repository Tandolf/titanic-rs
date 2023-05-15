use csv::{Reader, Writer};
use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::basic::matrix::DenseMatrix,
};

fn main() {
    let mut reader = Reader::from_path("data/train.csv").unwrap();

    let mut raw_data = Vec::with_capacity(100);
    let mut target = Vec::with_capacity(100);
    let mut sex_data = Vec::with_capacity(100);

    for result in reader.records() {
        let record = result.unwrap();
        let survived = record.get(1).unwrap();
        let sex = record.get(4).unwrap();
        let sex_binary = if sex == "female" { 1 } else { 0 };
        sex_data.push(sex_binary);

        let s = survived.parse::<u32>().unwrap();

        target.push(s);
        raw_data.push(record);
    }

    let features = vec!["Pclass", "Sex", "SibSp", "Parch"];

    let mut data = Vec::with_capacity(raw_data.len());

    for r in raw_data.iter() {
        let mut row = Vec::with_capacity(features.len() - 1);
        let p_class = r.get(2).unwrap();
        let sex = r.get(4).unwrap();
        let sex = if sex == "female" { 0. } else { 1. };
        let sib_sp = r.get(6).unwrap();
        let parch = r.get(7).unwrap();

        let p_class = p_class.parse::<f64>().unwrap();
        let sib_sp = sib_sp.parse::<f64>().unwrap();
        let parch = parch.parse::<f64>().unwrap();

        row.push(p_class);
        row.push(sex);
        row.push(sib_sp);
        row.push(parch);
        data.push(row);
    }

    let matrix = DenseMatrix::from_2d_vec(&data);

    let params = RandomForestClassifierParameters::default()
        .with_n_trees(500)
        .with_m(1)
        .with_max_depth(5);
    let classifier = RandomForestClassifier::fit(&matrix, &sex_data, params).unwrap();

    let mut reader = Reader::from_path("data/test.csv").unwrap();

    let mut data = Vec::with_capacity(raw_data.len());

    for result in reader.records() {
        let record = result.unwrap();

        let mut row = Vec::with_capacity(features.len() - 1);
        println!("{:?}", record);
        let p_class = record.get(1).unwrap();
        let sex = record.get(3).unwrap();
        let sex = if sex == "female" { 0. } else { 1. };
        let sib_sp = record.get(5).unwrap();
        let parch = record.get(6).unwrap();

        let p_class = p_class.parse::<f64>().unwrap();
        let sib_sp = sib_sp.parse::<f64>().unwrap();
        let parch = parch.parse::<f64>().unwrap();

        row.push(p_class);
        row.push(sex);
        row.push(sib_sp);
        row.push(parch);

        data.push(row);
    }

    let matrix = DenseMatrix::from_2d_vec(&data);

    let y_hat = classifier.predict(&matrix).unwrap();

    let mut wtr = Writer::from_path("submission.csv").unwrap();
    wtr.write_record(["PassengerId", "Survived"]).unwrap();
    for (i, s) in y_hat.iter().enumerate() {
        wtr.write_record([(i + 892).to_string(), s.to_string()])
            .unwrap();
    }

    wtr.flush().unwrap();
}
