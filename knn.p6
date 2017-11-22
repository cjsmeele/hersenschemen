use v6;
use lib '.';
use CSV;

sub MAIN(Str $dataset-file = 'dataset1.csv') {
    for (CSV::parse($dataset-file.IO.lines)) {
        say join ', ', @$_ for $_;
    }
}
