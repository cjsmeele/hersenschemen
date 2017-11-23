use v6;
use lib '.';
use CSV;

sub infix:<fl>($a, $b) {|($a, $b)};

sub euclidian_distance(@p1, @p2) {
    sqrt([+] map((*+*)**2, (@p1 Zfl @p2)));
}

sub MAIN(Str $dataset-file = 'dataset1.csv') {
    my @points = CSV::parse($dataset-file.IO.lines);
    .say for @points;
    
    my @arr1 = 1,2,3;
    my @arr2 = 4,5,6;
    say @arr1 Z @arr2;
    say @arr1 Zfl @arr2; #meta operators yo
    say euclidian_distance(@arr1, @arr2);
}
