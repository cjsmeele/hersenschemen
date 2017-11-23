use v6;
use lib '.';
use CSV;

sub euclidian_distance(@p1, @p2) {
    sqrt([+] map((*+*)**2, @p1 Z @p2));
}

sub MAIN(Str $dataset-file = 'dataset1.csv') {
    my @points = CSV::parse($dataset-file.IO.lines);
    .say for @points;
}
