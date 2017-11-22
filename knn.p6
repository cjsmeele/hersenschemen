use v6;
use lib '.';
use CSV;

sub MAIN(Str $dataset-file = 'dataset1.csv') {
    my @points = CSV::parse($dataset-file.IO.lines);
    .say for @points;
}
