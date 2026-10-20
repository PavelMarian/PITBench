#!/usr/bin/env bash
# Download / generate every raw input of PITBench into raw/<dataset>/.
# Usage: PITBULL_REPO=/path/to/pitbull ./fetch_raw.sh
# Needs: curl, git (for Olist from the project's history), tpchgen-cli (pip install tpchgen-cli).
set -euo pipefail
cd "$(dirname "$0")"
R=raw
dl() { mkdir -p "$R/$1"; curl -sSfL -m 600 -o "$R/$1/$2" "$3"; }

# Olist: the copy kept in the PITBULL history
mkdir -p $R/olist
for f in $(git -C "$PITBULL_REPO" ls-tree --name-only a9cd898 PITFALL_olist_data/); do
  git -C "$PITBULL_REPO" show "a9cd898:$f" > "$R/olist/$(basename "$f")"
done

dl f1 f1db-csv.zip https://github.com/f1db/f1db/releases/latest/download/f1db-csv.zip

for f in account card client disp district loan order trans; do
  dl pkdd99 $f.asc https://raw.githubusercontent.com/jlacko/berka-dataset/master/$f.asc
done

AW=https://raw.githubusercontent.com/microsoft/sql-server-samples/master/samples/databases/adventure-works/oltp-install-script
for f in SalesOrderHeader SalesOrderDetail Customer Product ProductListPriceHistory ProductCostHistory \
         ProductSubcategory ProductCategory SpecialOffer SalesTerritory SalesPerson; do
  dl adventureworks $f.csv $AW/$f.csv
done

NW=https://raw.githubusercontent.com/graphql-compose/graphql-compose-examples/master/examples/northwind/data/csv
for f in categories customers employees order_details orders products shippers suppliers; do
  dl northwind $f.csv $NW/$f.csv
done

dl sakila schema.sql https://raw.githubusercontent.com/jOOQ/sakila/main/sqlite-sakila-db/sqlite-sakila-schema.sql
dl sakila data.sql https://raw.githubusercontent.com/jOOQ/sakila/main/sqlite-sakila-db/sqlite-sakila-insert-data.sql

for f in ratings.dat movies.dat users.dat; do
  dl movietweetings $f https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/$f
done

mkdir -p $R/tpch && tpchgen-cli csv -s 0.05 --output-dir=$R/tpch

echo "raw inputs ready; compare with datasets/*/build_info.json (raw_sha256) for the exact versions used."
