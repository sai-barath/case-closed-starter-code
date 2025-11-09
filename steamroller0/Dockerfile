FROM golang:1.23-alpine AS builder

WORKDIR /app

COPY go.mod ./
RUN go mod download || true

COPY *.go ./

RUN go build -o steamroller .

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/steamroller .

ENV PORT=8080

EXPOSE 8080

CMD ["./steamroller"]
