use v6;
use lib '.';
use CSV;

sub euclidian-distance(@p1, @p2) {
    (@p1 Z- @p2).map(* **2).sum.sqrt;
}

sub infix:<|\<---\>|>(@p1, @p2) {
    euclidian-distance(@p1, @p2);
}

sub MAIN(Str $dataset-file = 'dataset1.csv') {
    say (1, 1, 1) |<--->| (0, 0, 0);
    return;
    my @points = CSV::parse($dataset-file.IO.lines);
    .say for @points;
}
