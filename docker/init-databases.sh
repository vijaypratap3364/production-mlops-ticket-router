#!/bin/sh
set -eu

validate_identifier() {
    case "$1" in
        ""|*[!a-zA-Z0-9_]*)
            echo "Database identifiers may contain only letters, numbers, and underscores." >&2
            exit 1
            ;;
    esac
}

create_database_if_missing() {
    database_name="$1"
    validate_identifier "$database_name"
    validate_identifier "$POSTGRES_USER"
    exists="$(
        psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --tuples-only --no-align \
            --command "SELECT 1 FROM pg_database WHERE datname = '$database_name'"
    )"
    if [ "$exists" != "1" ]; then
        psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
            --command "CREATE DATABASE \"$database_name\" OWNER \"$POSTGRES_USER\""
    fi
}

create_database_if_missing "${MLFLOW_POSTGRES_DB:-mlflow}"
create_database_if_missing "${PREFECT_POSTGRES_DB:-prefect}"
