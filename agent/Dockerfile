FROM alpine:3.21.3 AS builder

WORKDIR /build

COPY . .

RUN cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --parallel 8

FROM alpine:3.21.3 AS final

WORKDIR /app

COPY --from=builder quant .

ENTRYPOINT ["/app/quant"]